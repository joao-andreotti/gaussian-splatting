#!/bin/bash

# usage: ./process.sh  [video_path] [output_path] [images_path] [sample_size]

VIDEO_PATH=$1
OUTPUT_PATH=$2
IMAGES_PATH=$3
SAMPLE_SIZE=$4

# Check if the source and destination folders are provided
if [ -z "$VIDEO_PATH" ]; then
    echo "Video path not provided"
    exit 1
fi
if [ -z "$OUTPUT_PATH" ]; then
    echo "Output path not provided"
    exit 1
fi
if [ -z "$IMAGES_PATH" ]; then
    echo "Images path not provided"
    exit 1
fi
if [ -z "$SAMPLE_SIZE" ]; then
    echo "Images path not provided"
    exit 1
fi

mkdir _tmp;
mkdir _tmp/images;
mkdir _tmp/sample_images;
mkdir $OUTPUT_PATH;

# extracting frames from  video
ffmpeg -i "$VIDEO_PATH" ./_tmp/images/%04d.jpg;

# sampling images from folder
find ./_tmp/images -type f | shuf -n "$SAMPLE_SIZE" | xargs -I {} cp {} ./_tmp/sample_images

# running colmap
colmap automatic_reconstructor \
    --workspace_path ./_tmp \
    --image_path ./_tmp/sample_images \
    --camera_model SIMPLE_RADIAL;

# converting colmap output
colmap model_converter \
    --input_path ./_tmp/sparse/0 \
    --output_path ./$OUTPUT_PATH/ \
    --output_type TXT;

# copying the images to the desired folder
mkdir $IMAGES_PATH;
cp _tmp/images/* $IMAGES_PATH;

# cleanup
rm -r _tmp;

