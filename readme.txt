RUN: python download_model.py 

export SAPIENS_ROOT=/mnt/data/jing/Video_Generation/video_data_repos/video_smplx_labeling/sapiens
export SAPIENS_LITE_ROOT=$SAPIENS_ROOT/lite
export SAPIENS_LITE_CHECKPOINT_ROOT=$SAPIENS_ROOT/sapiens_lite_host

conda activate smplx_labeler

cd $SAPIENS_LITE_ROOT/scripts/demo/torchscript

cd $SAPIENS_LITE_ROOT/scripts/infer/torchscript
./pose_keypoints133.sh VIDEO_DIR device batchsize jobs_per_gpu

