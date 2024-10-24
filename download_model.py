from huggingface_hub import hf_hub_download

repo_id = "facebook/sapiens-pose-bbox-detector"
file_path = "rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
local_file_path = hf_hub_download(repo_id=repo_id, filename=file_path, local_dir="./sapiens_lite_host/torchscript/detector/checkpoints/rtmpose")
print(f"File downloaded to: {local_file_path}")


repo_id = "noahcao/sapiens-pose-coco"
file_path = "sapiens_lite_host/torchscript/pose/checkpoints/sapiens_2b/sapiens_2b_coco_wholebody_best_coco_wholebody_AP_745_torchscript.pt2"
local_file_path = hf_hub_download(repo_id=repo_id, filename=file_path, local_dir="./")
print(f"File downloaded to: {local_file_path}")