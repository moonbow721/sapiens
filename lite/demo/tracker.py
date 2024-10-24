import numpy as np
import torch
import cv2

from tqdm import tqdm
from collections import defaultdict

from ultralytics import YOLO

from tracker_utils import (
    get_frame_id_list_from_mask,
    linear_interpolate_frame_ids,
    frame_id_to_mask,
    rearrange_by_mask,
    moving_average_smooth,
    get_bbx_xys_from_xyxy_batch
)

def get_video_lwh(video_path):
    cap = cv2.VideoCapture(video_path)
    L = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return L, W, H


class Tracker:
    def __init__(self, yolo_path='yolov8x.pt') -> None:
        # https://docs.ultralytics.com/modes/predict/
        self.yolo = YOLO(yolo_path)

    def track(self, video_path):
        track_history = []
        cfg = {
            "device": "cuda",
            "conf": 0.5,  # default 0.25, wham 0.5
            "classes": 0,  # human
            "verbose": False,
            "stream": True,
        }
        results = self.yolo.track(video_path, **cfg)
        # frame-by-frame tracking
        track_history = []
        for result in tqdm(results, total=get_video_lwh(video_path)[0], desc="YoloV8 Tracking"):
            if result.boxes.id is not None:
                track_ids = result.boxes.id.int().cpu().tolist()  # (N)
                bbx_xyxy = result.boxes.xyxy.cpu().numpy()  # (N, 4)
                bbx_conf = result.boxes.conf.cpu().numpy()  # (N)
                result_frame = [{"id": track_ids[i], "bbx_xyxy": bbx_xyxy[i], "conf": bbx_conf[i]} for i in range(len(track_ids))]
            else:
                result_frame = []
            track_history.append(result_frame)

        return track_history

    @staticmethod
    def sort_track_length(track_history, video_path):
        """This handles the track history from YOLO tracker."""
        id_to_frame_ids = defaultdict(list)
        id_to_bbx_xyxys = defaultdict(list)
        id_to_confs = defaultdict(list)
        # parse to {det_id : [frame_id]}
        for frame_id, frame in enumerate(track_history):
            for det in frame:
                id_to_frame_ids[det["id"]].append(frame_id)
                id_to_bbx_xyxys[det["id"]].append(det["bbx_xyxy"])
                id_to_confs[det["id"]].append(det["conf"])
        for k, v in id_to_bbx_xyxys.items():
            id_to_bbx_xyxys[k] = np.array(v)
        for k, v in id_to_confs.items():
            id_to_confs[k] = np.array(v)

        # Sort by length of each track (max to min)
        id_length = {k: len(v) for k, v in id_to_frame_ids.items()}
        id2length = dict(sorted(id_length.items(), key=lambda item: item[1], reverse=True))

        # Sort by area sum (max to min)
        id_area_sum = {}
        l, w, h = get_video_lwh(video_path)
        for k, v in id_to_bbx_xyxys.items():
            bbx_wh = v[:, 2:] - v[:, :2]
            id_area_sum[k] = (bbx_wh[:, 0] * bbx_wh[:, 1] / w / h).sum()
        id2area_sum = dict(sorted(id_area_sum.items(), key=lambda item: item[1], reverse=True))
        id_sorted = list(id2area_sum.keys())

        return id_to_frame_ids, id_to_bbx_xyxys, id_to_confs, id_sorted

    def get_one_track(self, video_path):
        # track
        track_history = self.track(video_path)

        # parse track_history & use top1 track
        id_to_frame_ids, id_to_bbx_xyxys, id_to_confs, id_sorted = self.sort_track_length(track_history, video_path)
        track_id = id_sorted[0]
        frame_ids = torch.tensor(id_to_frame_ids[track_id])  # (N,)
        bbx_xyxys = torch.tensor(id_to_bbx_xyxys[track_id])  # (N, 4)
        confs = torch.tensor(id_to_confs[track_id])  # (N,)

        # interpolate missing frames
        mask = frame_id_to_mask(frame_ids, get_video_lwh(video_path)[0])
        bbx_xyxy_one_track = rearrange_by_mask(bbx_xyxys, mask)  # (F, 4), missing filled with 0
        conf_one_track = rearrange_by_mask(confs, mask)  # (F,), missing filled with 0
        missing_frame_id_list = get_frame_id_list_from_mask(~mask)  # list of list
        bbx_xyxy_one_track = linear_interpolate_frame_ids(bbx_xyxy_one_track, missing_frame_id_list)
        conf_one_track = linear_interpolate_frame_ids(conf_one_track, missing_frame_id_list)
        assert (bbx_xyxy_one_track.sum(1) != 0).all()

        bbx_xyxy_one_track = moving_average_smooth(bbx_xyxy_one_track, window_size=5, dim=0)
        bbx_xyxy_one_track = moving_average_smooth(bbx_xyxy_one_track, window_size=5, dim=0)
        conf_one_track = moving_average_smooth(conf_one_track, window_size=5, dim=0)

        return bbx_xyxy_one_track, conf_one_track

    def get_all_tracks(self, video_path, frame_thres=0.5):
        # Track all objects in the video
        track_history = self.track(video_path)

        # Parse track_history & use all tracks
        # import ipdb; ipdb.set_trace()
        id_to_frame_ids, id_to_bbx_xyxys, id_to_confs, id_sorted = self.sort_track_length(track_history, video_path)
        
        all_tracks = []
        all_confs = []
        
        for track_id in id_sorted:
            frame_ids = torch.tensor(id_to_frame_ids[track_id])  # (N,)
            if len(frame_ids) < frame_thres * get_video_lwh(video_path)[0]:
                continue
            bbx_xyxys = torch.tensor(id_to_bbx_xyxys[track_id])  # (N, 4)
            confs = torch.tensor(id_to_confs[track_id])  # (N,)

            # Interpolate missing frames
            mask = frame_id_to_mask(frame_ids, get_video_lwh(video_path)[0])
            bbx_xyxy_one_track = rearrange_by_mask(bbx_xyxys, mask)  # (F, 4), missing filled with 0
            conf_one_track = rearrange_by_mask(confs, mask)  # (F,), missing filled with 0
            missing_frame_id_list = get_frame_id_list_from_mask(~mask)  # list of list
            bbx_xyxy_one_track = linear_interpolate_frame_ids(bbx_xyxy_one_track, missing_frame_id_list)
            conf_one_track = linear_interpolate_frame_ids(conf_one_track, missing_frame_id_list)
            assert (bbx_xyxy_one_track.sum(1) != 0).all()

            bbx_xyxy_one_track = moving_average_smooth(bbx_xyxy_one_track, window_size=5, dim=0)
            bbx_xyxy_one_track = moving_average_smooth(bbx_xyxy_one_track, window_size=5, dim=0)
            conf_one_track = moving_average_smooth(conf_one_track, window_size=5, dim=0)

            # Append the processed track to the list
            all_tracks.append(bbx_xyxy_one_track)
            all_confs.append(conf_one_track)

        return torch.stack(all_tracks).float(), torch.stack(all_confs).float()  # (person_num, N, 4), (person_num, N)
    

if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser(description="Track objects in a video")
    parser.add_argument("--video_path", type=str, help="Path to the input video file")
    parser.add_argument("--save_path", type=str, help="Path to save the output .pt file")
    parser.add_argument("--frame_thres", type=float, default=0.5, help="Threshold for the number of frames to keep a track")
    parser.add_argument("--yolo_checkpoint", type=str, help="Path to the YOLO checkpoint file")
    args = parser.parse_args()

    tracker = Tracker(args.yolo_checkpoint)
    bbx_xyxy, bbx_conf = tracker.get_all_tracks(args.video_path, args.frame_thres)
    bbx_xys = get_bbx_xys_from_xyxy_batch(bbx_xyxy, base_enlarge=1.2).float()
    
    # Save the results
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save({
        'bbx_xyxy': bbx_xyxy.detach().cpu(),
        'bbx_conf': bbx_conf.detach().cpu(),
        'bbx_xys': bbx_xys.detach().cpu()
    }, args.save_path)