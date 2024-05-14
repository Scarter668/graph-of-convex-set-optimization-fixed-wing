#!/bin/bash

# Function to print usage information
usage() {
    echo "Usage: $0 [--folder=<target_folder>] [--name=<output_video_name>]"
    exit 1
}

# Parse command-line arguments
TARGET_FOLDER= $PWD
OUTPUT_VIDEO_NAME="output.mp4"
TAR_FILE=""

echo $TARGET_FOLDER

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --folder=*) TARGET_FOLDER="${1#*=}" ;;
        --name=*) OUTPUT_VIDEO_NAME="${1#*=}" ;;
        *) usage ;;
    esac
    shift
done

# Use the current directory if TARGET_FOLDER is not specified
if [[ -z "$TARGET_FOLDER" ]]; then
    TARGET_FOLDER="."
fi


# Navigate to the target folder
cd "$TARGET_FOLDER" || { echo "Could not navigate to folder: $TARGET_FOLDER"; exit 1; }

# Find the first tar file if one is not specified
TAR_FILE=$(find . -maxdepth 1 -name "*.tar*" | head -n 1)

if [[ -z "$TAR_FILE" ]]; then
    echo "Error: No tar file found in the specified folder."
    exit 1
fi

# Extract the tar file
tar -xf "$TAR_FILE"

# Check if there are any PNG files to process
if ls *.png 1> /dev/null 2>&1; then
    # Run ffmpeg to create the video from the PNG images
    ffmpeg -r 60 -i %07d.png -vcodec libx264 -preset slow -crf 18 "$OUTPUT_VIDEO_NAME".mp4
    
    # Remove all the PNG files after the video is created
    rm -f *.png
else
    echo "No PNG files found after extraction."
    exit 1
fi
