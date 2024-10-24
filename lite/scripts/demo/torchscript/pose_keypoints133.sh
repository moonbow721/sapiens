#!/bin/bash

cd ../../.. || exit
SAPIENS_CHECKPOINT_ROOT=${SAPIENS_LITE_CHECKPOINT_ROOT}

MODE='torchscript' ## original. no optimizations (slow). full precision inference.
# MODE='bfloat16' ## A100 gpus. faster inference at bfloat16

SAPIENS_CHECKPOINT_ROOT=$SAPIENS_CHECKPOINT_ROOT/$MODE

#----------------------------set your input and output directories----------------------------------------------
INPUT_VIDEO='/mnt/data/jing/Video_Generation/video_data_repos/video_smplx_labeling/sapiens/example_data1/two_persons.mp4'
VIDEO_DIR=$(dirname "$INPUT_VIDEO")
FILE_NAME=$(basename "$INPUT_VIDEO" .mp4)
FRAMES_DIR="${VIDEO_DIR}/frames"

mkdir -p ${FRAMES_DIR}
ffmpeg -i ${INPUT_VIDEO} ${FRAMES_DIR}/frame_%06d.jpg

INPUT=${FRAMES_DIR}
OUTPUT="${VIDEO_DIR}/${FILE_NAME}"

# INPUT='/mnt/data/jing/Video_Generation/video_data_repos/video_smplx_labeling/sapiens/example_data3/images'
# OUTPUT='/mnt/data/jing/Video_Generation/video_data_repos/video_smplx_labeling/sapiens/example_data3/results_pose133'

#--------------------------MODEL CARD---------------
# MODEL_NAME='sapiens_0.3b'; CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/pose/checkpoints/sapiens_0.3b/sapiens_0.3b_coco_wholebody_best_coco_wholebody_AP_620_$MODE.pt2
# MODEL_NAME='sapiens_0.6b'; CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/pose/checkpoints/sapiens_0.6b/sapiens_0.6b_coco_wholebody_best_coco_wholebody_AP_695_$MODE.pt2
# MODEL_NAME='sapiens_1b'; CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/pose/checkpoints/sapiens_1b/sapiens_1b_coco_wholebody_best_coco_wholebody_AP_727_$MODE.pt2
MODEL_NAME='sapiens_2b'; CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/pose/checkpoints/sapiens_2b/sapiens_2b_coco_wholebody_best_coco_wholebody_AP_745_$MODE.pt2

OUTPUT=$OUTPUT/$MODEL_NAME

DETECTION_CONFIG_FILE='../pose/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person_no_nms.py'
DETECTION_CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/detector/checkpoints/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth

#---------------------------VISUALIZATION PARAMS--------------------------------------------------
LINE_THICKNESS=1 ## line thickness of the skeleton
RADIUS=2 ## keypoint radius
KPT_THRES=0.3 ## confidence threshold

##-------------------------------------inference-------------------------------------
RUN_FILE='demo/vis_pose.py'

## number of inference jobs per gpu, total number of gpus and gpu ids
# JOBS_PER_GPU=1; TOTAL_GPUS=8; VALID_GPU_IDS=(0 1 2 3 4 5 6 7)
JOBS_PER_GPU=2; TOTAL_GPUS=1; VALID_GPU_IDS=(7)

BATCH_SIZE=32

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
    --num_keypoints 133 \
    --det-config ${DETECTION_CONFIG_FILE} \
    --det-checkpoint ${DETECTION_CHECKPOINT} \
    --batch-size ${BATCH_SIZE} \
    --input "${INPUT}/image_paths_$((i+1)).txt" \
    --output-root="${OUTPUT}" \
    --radius ${RADIUS} \
    --thickness ${LINE_THICKNESS} \
    --kpt-thr ${KPT_THRES} & ## add & to process in background
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

# Go back to the original script's directory
cd -

echo "Processing complete."
echo "Results saved to $OUTPUT"

# Get the frame rate from the input video
INPUT_FPS=$(ffprobe -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=r_frame_rate "${INPUT_VIDEO}")
# Convert fraction to decimal if necessary
INPUT_FPS=$(echo "scale=2; $INPUT_FPS" | bc)

# Merge the saved visualization frames into a video through ffmpeg
if [ -d "${OUTPUT}" ]; then
    echo "Merging visualization frames into a video..."
    ffmpeg -framerate ${INPUT_FPS} -pattern_type glob -i "${OUTPUT}/*.jpg" -c:v libx264 -pix_fmt yuv420p "${VIDEO_DIR}/visualization.mp4"
    echo "Video saved as ${VIDEO_DIR}/visualization.mp4"
else
    echo "No visualization frames found. Skipping video creation."
fi