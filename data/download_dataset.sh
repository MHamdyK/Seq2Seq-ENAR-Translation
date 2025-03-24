#!/bin/bash

# download_dataset.sh
# This script downloads the dataset.zip from Google Drive and extracts it.

# Check if gdown is installed; if not, install it.
if ! command -v gdown &> /dev/null
then
    echo "gdown is not installed. Installing gdown..."
    pip install gdown
fi

# Google Drive file ID and output filename
FILEID="1MNgjbLi7mB1wmX9SwL3JOdjBZjjM7VSi"
FILENAME="Dataset.zip"

echo "Downloading dataset from Google Drive..."
gdown --id $FILEID -O $FILENAME

if [ $? -ne 0 ]; then
    echo "Error: Download failed."
    exit 1
fi

echo "Download completed successfully!"

# Define extraction directory (change if needed)
EXTRACT_DIR="/content"

echo "Extracting dataset to ${EXTRACT_DIR}..."
unzip -o $FILENAME -d $EXTRACT_DIR

if [ $? -ne 0 ]; then
    echo "Error: Extraction failed."
    exit 1
fi

echo "Dataset extracted successfully!"
