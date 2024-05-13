#!/bin/bash

# Set HOME variable and print it
HOME=$(pwd)
echo "HOME: $HOME"

# Install necessary Python packages
pip3 install -q 'git+https://github.com/facebookresearch/segment-anything.git'
pip3 install -q jupyter_bbox_widget roboflow dataclasses-json supervision

# Create directory for weights and download the model weights
mkdir -p ${HOME}/sam-weights
wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P ${HOME}/sam-weights

# Check if the model checkpoint exists
CHECKPOINT_PATH="${HOME}/sam-weights/sam_vit_h_4b8939.pth"
if [ -f "$CHECKPOINT_PATH" ]; then
    echo "$CHECKPOINT_PATH ; exist: true"
else
    echo "$CHECKPOINT_PATH ; exist: false"
fi

# Create directory for test images and download images
mkdir -p ${HOME}/data/test_images
wget -q https://media.roboflow.com/notebooks/examples/dog.jpeg -P ${HOME}/data/test_images
wget -q https://media.roboflow.com/notebooks/examples/dog-2.jpeg -P ${HOME}/data/test_images
wget -q https://media.roboflow.com/notebooks/examples/dog-3.jpeg -P ${HOME}/data/test_images
wget -q https://media.roboflow.com/notebooks/examples/dog-4.jpeg -P ${HOME}/data/test_images

echo "Setup completed."
