#!/bin/bash

# usage: ./process_synthetic.sh  [snynthetic_files_path] [dataset_name] [output_path]

DATASET_PATH=$1 #nerf/NeRF_Data/
DATASET=$2
INPUT_PATH="$DATASET_PATH/$DATASET"
OUTPUT_PATH=$3
IMAGES_PATH="$INPUT_PATH/train"

mkdir _tmp;
mkdir $OUTPUT_PATH;
mkdir "$OUTPUT_PATH/images";

cp -r "$IMAGES_PATH" ./_tmp/

# running colmap
colmap automatic_reconstructor \
    --workspace_path ./_tmp \
    --image_path ./_tmp/train\
    --camera_model PINHOLE;

# undistoring camera parameters
colmap image_undistorter \
    --image_path ./_tmp/train \
    --input_path ./_tmp/sparse/0 \
    --output_path ./_tmp/dense \
    --output_type COLMAP;

# converting colmap output
colmap model_converter \
    --input_path ./_tmp/dense/sparse/ \
    --output_path ./$OUTPUT_PATH/ \
    --output_type TXT;

cp -r ./_tmp/dense/images/* "$OUTPUT_PATH/images";

rm -r _tmp;

echo "Done!"
