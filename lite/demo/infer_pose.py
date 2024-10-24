# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import multiprocessing as mp
import os
import time
import warnings
from argparse import ArgumentParser
from collections import defaultdict
from functools import partial
from multiprocessing import cpu_count, Pool, Process
from typing import List, Optional, Sequence, Union

import cv2
import json_tricks as json
import torch
import torch.nn as nn
import torch.nn.functional as F
from adhoc_image_dataset import AdhocImageDataset
from pose_utils import nms, top_down_affine_transform, udp_decode

from tqdm import tqdm

from worker_pool import WorkerPool


warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=UserWarning, module="mmengine")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")
warnings.filterwarnings("ignore", category=UserWarning, module="json_tricks.encoders")

timings = {}
BATCH_SIZE = 48


def preprocess_pose(orig_img, bboxes_list, input_shape, mean, std):
    """Preprocess pose images and bboxes."""
    preprocessed_images = []
    centers = []
    scales = []
    for bbox in bboxes_list:
        img, center, scale = top_down_affine_transform(orig_img.copy(), bbox)
        img = cv2.resize(
            img, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_LINEAR
        ).transpose(2, 0, 1)
        img = torch.from_numpy(img)
        img = img[[2, 1, 0], ...].float()
        mean = torch.Tensor(mean).view(-1, 1, 1)
        std = torch.Tensor(std).view(-1, 1, 1)
        img = (img - mean) / std
        preprocessed_images.append(img)
        centers.extend(center)
        scales.extend(scale)
    return preprocessed_images, centers, scales


def batch_inference_topdown(
    model: nn.Module,
    imgs: List[Union[np.ndarray, str]],
    dtype=torch.bfloat16,
    flip=False,
):
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
        heatmaps = model(imgs.cuda())
        if flip:
            heatmaps_ = model(imgs.to(dtype).cuda().flip(-1))
            heatmaps = (heatmaps + heatmaps_) * 0.5
        imgs.cpu()
    return heatmaps.cpu()


def save_results(results, output_path, input_shape, heatmap_scale):
    heatmap = results["heatmaps"]
    centres = results["centres"]
    scales = results["scales"]
    instance_keypoints = []
    instance_scores = []
    for i in range(len(heatmap)):
        result = udp_decode(
            heatmap[i].cpu().unsqueeze(0).float().data[0].numpy(),
            input_shape,
            (int(input_shape[0] / heatmap_scale), int(input_shape[1] / heatmap_scale)),
        )

        keypoints, keypoint_scores = result
        keypoints = (keypoints / input_shape) * scales[i] + centres[i] - 0.5 * scales[i]
        instance_keypoints.append(keypoints[0])
        instance_scores.append(keypoint_scores[0])

    with open(output_path, "w") as f:
        json.dump(
            dict(
                instance_info=[
                    {
                        "keypoints": keypoints.tolist(),
                        "keypoint_scores": keypoint_scores.tolist(),
                    }
                    for keypoints, keypoint_scores in zip(
                        instance_keypoints, instance_scores
                    )
                ]
            ),
            f,
            indent="\t",
        )

def fake_pad_images_to_batchsize(imgs):
    return F.pad(imgs, (0, 0, 0, 0, 0, 0, 0, BATCH_SIZE - imgs.shape[0]), value=0)

def load_model(checkpoint, use_torchscript=False):
    if use_torchscript:
        return torch.jit.load(checkpoint)
    else:
        return torch.export.load(checkpoint).module()

def main():
    """Visualize the demo images.
    Using YOLO to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument("pose_checkpoint", help="Checkpoint file for pose")
    parser.add_argument("--input", type=str, default="", help="Image/Video file")
    parser.add_argument("--track-result-path", type=str, default="", help="Path to the track result file")
    parser.add_argument(
        "--shape",
        type=int,
        nargs="+",
        default=[1024, 768],
        help="input image size (height, width)",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="",
        help="root of the output img file. "
        "Default not saving the visualization images.",
    )
    parser.add_argument(
        "--batch_size",
        "--batch-size",
        type=int,
        default=48,
        help="Set batch size to do batch inference. ",
    )
    parser.add_argument(
        "--fp16", action="store_true", default=False, help="Model inference dtype"
    )
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument(
        "--heatmap-scale", type=int, default=4, help="Heatmap scale for keypoints. Image to heatmap ratio"
    )
    parser.add_argument(
        "--flip",
        type=bool,
        default=False,
        help="Flip the input image horizontally and inference again",
    )

    args = parser.parse_args()
    assert args.input != "", "Please provide the input file."

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3,) + tuple(args.shape)
    else:
        raise ValueError("invalid input shape")

    mp.log_to_stderr()
    import torch._inductor.config as config
    config.force_fuse_int_mm_with_mul = True
    config.use_mixed_mm = True

    start = time.time()

    if not os.path.exists(args.output_root):
        os.makedirs(args.output_root)

    assert args.output_root != ""
    args.pred_save_path = (
        f"{args.output_root}/results_"
        f"{os.path.splitext(os.path.basename(args.input))[0]}.json"
    )

    # build pose estimator
    USE_TORCHSCRIPT = '_torchscript' in args.pose_checkpoint

    # build the model from a checkpoint file
    pose_estimator = load_model(args.pose_checkpoint, USE_TORCHSCRIPT)

    ## no precision conversion needed for torchscript. run at fp32
    if not USE_TORCHSCRIPT:
        dtype = torch.half if args.fp16 else torch.bfloat16
        pose_estimator.to(dtype)
        pose_estimator = torch.compile(pose_estimator, mode="max-autotune", fullgraph=True)
    else:
        dtype = torch.float32  # TorchScript models use float32
        pose_estimator = pose_estimator.to(args.device)

    global BATCH_SIZE
    BATCH_SIZE = args.batch_size

    input = args.input
    image_names = []

    # Check if the input is a directory or a text file
    if os.path.isdir(input):
        input_dir = input  # Set input_dir to the directory specified in input
        image_names = [
            image_name
            for image_name in sorted(os.listdir(input_dir))
            if image_name.endswith(".jpg") or image_name.endswith(".png")
        ]
    elif os.path.isfile(input) and input.endswith(".txt"):
        # If the input is a text file, read the paths from it and set input_dir to the directory of the first image
        with open(input, "r") as file:
            image_paths = [line.strip() for line in file if line.strip()]
        image_names = [
            os.path.basename(path) for path in image_paths
        ]  # Extract base names for image processing
        input_dir = (
            os.path.dirname(image_paths[0]) if image_paths else ""
        )  # Use the directory of the first image path

    scale = args.heatmap_scale
    inference_dataset = AdhocImageDataset(
        [os.path.join(input_dir, img_name) for img_name in image_names],
        args.track_result_path,
    )  # do not provide preprocess args for detector as we use mmdet
    inference_dataloader = torch.utils.data.DataLoader(
        inference_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(min(args.batch_size, cpu_count()) // 4, 4),
    )
    pose_preprocess_pool = WorkerPool(
        preprocess_pose, processes=max(min(args.batch_size, cpu_count()), 1)
    )
    
    for batch_idx, (batch_image_name, batch_orig_imgs, bboxes_batch) in tqdm(
        enumerate(inference_dataloader), total=len(inference_dataloader)
    ):

        valid_images_len = len(batch_orig_imgs)
        
        # import ipdb; ipdb.set_trace()

        assert len(bboxes_batch) == valid_images_len

        img_bbox_map = {}
        for i, bboxes in enumerate(bboxes_batch):
            img_bbox_map[i] = len(bboxes)

        args_list = [
            (
                i,
                bbox_list,
                (input_shape[1], input_shape[2]),
                [123.5, 116.5, 103.5],
                [58.5, 57.0, 57.5],
            )
            for i, bbox_list in zip(batch_orig_imgs.numpy(), bboxes_batch)
        ]
        pose_ops = pose_preprocess_pool.run(args_list)

        pose_imgs, pose_img_centers, pose_img_scales = [], [], []
        for op in pose_ops:
            pose_imgs.extend(op[0])
            pose_img_centers.extend(op[1])
            pose_img_scales.extend(op[2])

        n_pose_batches = (len(pose_imgs) + args.batch_size - 1) // args.batch_size

        # use this to tell torch compiler the start of model invocation as in 'flip' mode the tensor output is overwritten
        torch.compiler.cudagraph_mark_step_begin()  
        pose_results = []
        for i in range(n_pose_batches):
            imgs = torch.stack(
                pose_imgs[i * args.batch_size : (i + 1) * args.batch_size], dim=0
            )
            valid_len = len(imgs)
            imgs = fake_pad_images_to_batchsize(imgs)
            pose_results.extend(
                batch_inference_topdown(pose_estimator, imgs, dtype=dtype)[:valid_len]
            )

        batched_results = []
        for _, bbox_len in img_bbox_map.items():
            result = {
                "heatmaps": pose_results[:bbox_len].copy(),
                "centres": pose_img_centers[:bbox_len].copy(),
                "scales": pose_img_scales[:bbox_len].copy(),
            }
            batched_results.append(result)
            del (
                pose_results[:bbox_len],
                pose_img_centers[:bbox_len],
                pose_img_scales[:bbox_len],
            )

        assert len(batched_results) == len(batch_orig_imgs)

        for r, img_name in zip(
            batched_results[:valid_images_len],
            batch_image_name,
        ):
            output_path = os.path.join(args.output_root, os.path.basename(img_name).replace('.jpg', '.json').replace('.png', '.json'))
            save_results(r, output_path, (input_shape[2], input_shape[1]), scale)

    pose_preprocess_pool.finish()

    total_time = time.time() - start
    fps = 1 / ((time.time() - start) / len(image_names))
    print(
        f"\033[92mTotal inference time: {total_time:.2f} seconds. FPS: {fps:.2f}\033[0m"
    )


if __name__ == "__main__":
    main()
