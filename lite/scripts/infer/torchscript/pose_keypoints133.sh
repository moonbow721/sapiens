#!/bin/bash

VIDEO_DIR=$1
device_id=$2
batchsize=$3
jobs_per_gpu=$4

#----------------------------check if the results already exist----------------------------------------------
if [ -f "$VIDEO_DIR/sapiens_wholbody.pt" ]; then
    echo "Sapiens results already exist at $VIDEO_DIR/sapiens_wholbody.pt"
    exit 0
fi

cd ../../.. || exit
SAPIENS_CHECKPOINT_ROOT=${SAPIENS_LITE_CHECKPOINT_ROOT}

MODE='torchscript' ## original. no optimizations (slow). full precision inference.
# MODE='bfloat16' ## A100 gpus. faster inference at bfloat16
ENABLE_VIS=false
CLEAN_FRAMES=false

SAPIENS_CHECKPOINT_ROOT=$SAPIENS_CHECKPOINT_ROOT/$MODE

#----------------------------set your input and output directories----------------------------------------------
INPUT_VIDEO=$VIDEO_DIR/source_video.mp4
FRAMES_DIR="${VIDEO_DIR}/frames"

mkdir -p ${FRAMES_DIR}
ffmpeg -i ${INPUT_VIDEO} ${FRAMES_DIR}/frame_%06d.jpg

INPUT=${FRAMES_DIR}
OUTPUT=${VIDEO_DIR}

##-------------------------------------preprocess-------------------------------------
RUN_FILE='demo/tracker.py'
DETECTION_CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/detector/checkpoints/yolo/yolov8x.pt

TRACK_SAVE_PATH=${OUTPUT}/preprocess/bbx.pt

if [ ! -f "$TRACK_SAVE_PATH" ]; then
    CUDA_VISIBLE_DEVICES=${device_id} python ${RUN_FILE} \
        --video_path ${INPUT_VIDEO} \
        --save_path ${TRACK_SAVE_PATH} \
        --frame_thres 0.5 \
        --yolo_checkpoint ${DETECTION_CHECKPOINT}

    echo "Tracking results saved to ${TRACK_SAVE_PATH}"
else
    echo "Tracking results already exist at ${TRACK_SAVE_PATH}"
fi


#--------------------------MODEL CARD---------------
# MODEL_NAME='sapiens_0.3b'; CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/pose/checkpoints/sapiens_0.3b/sapiens_0.3b_coco_wholebody_best_coco_wholebody_AP_620_$MODE.pt2
# MODEL_NAME='sapiens_0.6b'; CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/pose/checkpoints/sapiens_0.6b/sapiens_0.6b_coco_wholebody_best_coco_wholebody_AP_695_$MODE.pt2
# MODEL_NAME='sapiens_1b'; CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/pose/checkpoints/sapiens_1b/sapiens_1b_coco_wholebody_best_coco_wholebody_AP_727_$MODE.pt2
MODEL_NAME='sapiens_2b'; CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/pose/checkpoints/sapiens_2b/sapiens_2b_coco_wholebody_best_coco_wholebody_AP_745_$MODE.pt2

KPTS_OUTPUT=$OUTPUT/$MODEL_NAME


##-------------------------------------inference-------------------------------------
RUN_FILE='demo/infer_pose.py'

## number of inference jobs per gpu, total number of gpus and gpu ids
## On our lambda machine, setting jobs x 2 is faster than batch_size x 2
# JOBS_PER_GPU=1; TOTAL_GPUS=8; VALID_GPU_IDS=(0 1 2 3 4 5 6 7)
JOBS_PER_GPU=$jobs_per_gpu; TOTAL_GPUS=1; VALID_GPU_IDS=($device_id)

BATCH_SIZE=$batchsize

# Find all images and sort them, then write to a temporary text file
IMAGE_LIST="${INPUT}/image_list.txt"
find "${INPUT}" -type f \( -iname \*.jpg -o -iname \*.png \) | sort > "${IMAGE_LIST}"

# Check if image list was created successfully
if [ ! -s "${IMAGE_LIST}" ]; then
  echo "No images found. Check your input directory and permissions."
  exit 1
fi

# Count images and calculate the number of images per text file
NUM_IMAGES=$(wc -l < "${IMAGE_LIST}")
if ((TOTAL_GPUS > NUM_IMAGES / BATCH_SIZE)); then
  TOTAL_JOBS=$(( (NUM_IMAGES + BATCH_SIZE - 1) / BATCH_SIZE))
  IMAGES_PER_FILE=$((BATCH_SIZE))
  EXTRA_IMAGES=$((NUM_IMAGES - ((TOTAL_JOBS - 1) * BATCH_SIZE)  ))
else
  TOTAL_JOBS=$((JOBS_PER_GPU * TOTAL_GPUS))
  IMAGES_PER_FILE=$((NUM_IMAGES / TOTAL_JOBS))
  EXTRA_IMAGES=$((NUM_IMAGES % TOTAL_JOBS))
fi

export TF_CPP_MIN_LOG_LEVEL=2
echo "Distributing ${NUM_IMAGES} image paths into ${TOTAL_JOBS} jobs."

# Divide image paths into text files for each job
for ((i=0; i<TOTAL_JOBS; i++)); do
  TEXT_FILE="${INPUT}/image_paths_$((i+1)).txt"
  if [ $i -eq $((TOTAL_JOBS - 1)) ]; then
    # For the last text file, write all remaining image paths
    tail -n +$((IMAGES_PER_FILE * i + 1)) "${IMAGE_LIST}" > "${TEXT_FILE}"
  else
    # Write the exact number of image paths per text file
    head -n $((IMAGES_PER_FILE * (i + 1))) "${IMAGE_LIST}" | tail -n ${IMAGES_PER_FILE} > "${TEXT_FILE}"
  fi
done

# Run the process on the GPUs, allowing multiple jobs per GPU
for ((i=0; i<TOTAL_JOBS; i++)); do
  GPU_ID=$((i % TOTAL_GPUS))
  CUDA_VISIBLE_DEVICES=${VALID_GPU_IDS[GPU_ID]} python ${RUN_FILE} \
    ${CHECKPOINT} \
    --track-result-path ${TRACK_SAVE_PATH} \
    --batch-size ${BATCH_SIZE} \
    --input "${INPUT}/image_paths_$((i+1)).txt" \
    --output-root="${KPTS_OUTPUT}" & ## add & to process in background
  # Allow a short delay between starting each job to reduce system load spikes
  sleep 1
done

# Wait for all background processes to finish
wait

# Remove the image list and temporary text files
rm "${IMAGE_LIST}"
for ((i=0; i<TOTAL_JOBS; i++)); do
  rm "${INPUT}/image_paths_$((i+1)).txt"
done

echo "Kpts results saved to $KPTS_OUTPUT"

##--------------------------------------CLEAN & VISUALIZATION----------------------------------
LINE_THICKNESS=2 ## line thickness of the skeleton
RADIUS=4 ## keypoint radius
KPT_THRES=0.3 ## confidence threshold

RUN_FILE='demo/infer_pose_cleaner.py'
KPTS_SAVE_PATH=${OUTPUT}/sapiens_wholebody.pt
VIS_SAVE_PATH=${OUTPUT}/sapiens_kpts_visualization.mp4

PYTHON_CMD="python ${RUN_FILE} \
    --video_path ${INPUT_VIDEO} \
    --keypoints_folder ${KPTS_OUTPUT} \
    --save_path ${KPTS_SAVE_PATH} \
    --output_video_path ${VIS_SAVE_PATH} \
    --num_keypoints 133 \
    --kpt_thr ${KPT_THRES} \
    --radius ${RADIUS} \
    --thickness ${LINE_THICKNESS}"

if ${ENABLE_VIS}; then
    PYTHON_CMD="$PYTHON_CMD --enable_vis"
fi

$PYTHON_CMD

echo "Merged kpts results saved to ${KPTS_SAVE_PATH}"
if ${ENABLE_VIS}; then
    echo "Visualization results saved to ${VIS_SAVE_PATH}"
fi

# Delete the FRAMES_DIR
if ${CLEAN_FRAMES}; then
    rm -rf ${FRAMES_DIR}
fi

# Go back to the original script's directory
cd -

echo "Sapiens wholebody keypoints processing complete."
