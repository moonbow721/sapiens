import argparse
import os
import numpy as np
import subprocess
import cv2
import json
import torch
from tqdm import tqdm
from classes_and_palettes import (
    COCO_KPTS_COLORS,
    COCO_WHOLEBODY_KPTS_COLORS,
    GOLIATH_KPTS_COLORS,
    GOLIATH_SKELETON_INFO,
    COCO_SKELETON_INFO,
    COCO_WHOLEBODY_SKELETON_INFO
)

def draw_keypoints(img, keypoints, keypoint_scores, kpt_colors, kpt_thr, radius, skeleton_info, thickness):
    for kid, (kpt, score) in enumerate(zip(keypoints, keypoint_scores)):
        if score < kpt_thr or kpt_colors[kid] is None:
            continue
        color = tuple(int(c) for c in kpt_colors[kid][::-1])
        cv2.circle(img, (int(kpt[0]), int(kpt[1])), radius, color, -1)
    
    for skid, link_info in skeleton_info.items():
        pt1_idx, pt2_idx = link_info['link']
        color = tuple(int(c) for c in link_info['color'][::-1])
        
        pt1 = keypoints[pt1_idx]
        pt2 = keypoints[pt2_idx]
        score1 = keypoint_scores[pt1_idx]
        score2 = keypoint_scores[pt2_idx]
        
        if score1 > kpt_thr and score2 > kpt_thr:
            cv2.line(img, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), color, thickness=thickness)
    
    return img

def main():
    parser = argparse.ArgumentParser(description="Merge json files and visualize pose on video")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--keypoints_folder", type=str, required=True, help="Path to folder containing keypoint JSON files")
    parser.add_argument("--save_path", type=str, required=True, help="Path to output keypoints .pt file")
    
    parser.add_argument("--enable_vis", action='store_true', help="Enable visualization")
    parser.add_argument("--output_video_path", type=str, default="./visualization.mp4", help="Path to output video")
    parser.add_argument("--num_keypoints", type=int, default=133, help="Number of keypoints")
    parser.add_argument("--kpt_thr", type=float, default=0.3, help="Keypoint score threshold")
    parser.add_argument("--radius", type=int, default=4, help="Keypoint radius for visualization")
    parser.add_argument("--thickness", type=int, default=2, help="Line thickness for skeleton")
    args = parser.parse_args()

    # Read all json files and merge into .pt tensor
    try:
        json_files = sorted([f for f in os.listdir(args.keypoints_folder) if f.endswith('.json')])
    except:
        json_files = []
        print(f"No JSON files found in {args.keypoints_folder}")
        
    
    if len(json_files) > 0:
        all_keypoints = []
        all_keypoint_scores = []
        
        for json_file in tqdm(json_files, desc="Processing JSON files"):
            with open(os.path.join(args.keypoints_folder, json_file), 'r') as f:
                data = json.load(f)
            
            frame_keypoints = []
            frame_keypoint_scores = []
            
            for instance in data['instance_info']:
                keypoints = np.array(instance['keypoints']).reshape(-1, 2)
                keypoint_scores = np.array(instance['keypoint_scores'])
                
                frame_keypoints.append(keypoints)
                frame_keypoint_scores.append(keypoint_scores)
            
            all_keypoints.append(frame_keypoints)
            all_keypoint_scores.append(frame_keypoint_scores)
        
        # Convert to tensor
        all_keypoints_tensor = torch.from_numpy(np.array(all_keypoints))
        all_keypoint_scores_tensor = torch.from_numpy(np.array(all_keypoint_scores))
        kpts = torch.cat([all_keypoints_tensor, all_keypoint_scores_tensor[..., None]], dim=-1)
        
        torch.save(kpts.transpose(0, 1), args.save_path)
        # # Save to .pt file
        # torch.save({
        #     'keypoints': all_keypoints_tensor,
        #     'keypoint_scores': all_keypoint_scores_tensor
        # }, args.save_path)
        
        print(f"Saved merged keypoints to {args.save_path}")
        
        # Delete the JSON folder
        try:
            subprocess.run(["rm", "-rf", args.keypoints_folder], check=True)
            print(f"Successfully deleted {args.keypoints_folder}")
        except:
            print(f"Failed to delete {args.keypoints_folder}")
    else:
        # Load tensor from save_path
        data = torch.load(args.save_path)
        all_keypoints_tensor = data[:, :, :, :2]  # (num_humans, num_frames, 133, 2)
        all_keypoint_scores_tensor = data[:, :, :, 2]  # (num_frames, num_humans, 133)
        print(f"Loaded keypoints from {args.save_path}")
    
    
    if args.enable_vis:
        # Set up keypoint colors and skeleton info based on number of keypoints
        if args.num_keypoints == 17:
            KPTS_COLORS = COCO_KPTS_COLORS
            SKELETON_INFO = COCO_SKELETON_INFO
        elif args.num_keypoints == 133:
            KPTS_COLORS = COCO_WHOLEBODY_KPTS_COLORS
            SKELETON_INFO = COCO_WHOLEBODY_SKELETON_INFO
        elif args.num_keypoints == 308:
            KPTS_COLORS = GOLIATH_KPTS_COLORS
            SKELETON_INFO = GOLIATH_SKELETON_INFO
        else:
            raise ValueError(f"Unsupported number of keypoints: {args.num_keypoints}")

        # Open video
        cap = cv2.VideoCapture(args.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output_video_path, fourcc, fps, (width, height))

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for frame_idx in tqdm(range(frame_count), desc="Visualizing frames"):
            ret, frame = cap.read()
            if not ret:
                break

            # Load corresponding keypoints from tensor
            keypoints = all_keypoints_tensor[:, frame_idx]
            keypoint_scores = all_keypoint_scores_tensor[:, frame_idx]

            # Draw keypoints and skeleton for each instance
            for instance_keypoints, instance_keypoint_scores in zip(keypoints, keypoint_scores):
                frame = draw_keypoints(frame, instance_keypoints.numpy(), instance_keypoint_scores.numpy(), KPTS_COLORS, args.kpt_thr, args.radius, SKELETON_INFO, args.thickness)

            out.write(frame)

        cap.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

'''
python infer_pose_cleaner.py --video_path /mnt/data/jing/Video_Generation/video_data_repos/video_smplx_labeling/sapiens/example_data1/two_persons.mp4 \
    --keypoints_folder /mnt/data/jing/Video_Generation/video_data_repos/video_smplx_labeling/sapiens/example_data1/two_persons/sapiens_2b \
    --save_path /mnt/data/jing/Video_Generation/video_data_repos/video_smplx_labeling/sapiens/example_data1/two_persons/kpts_2d_133.pt \
'''