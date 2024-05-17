import cv2
import torch

import numpy as np

from transformers import CLIPImageProcessor
from PIL import Image
from typing import List


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def normalize_bbox_coords(bboxes, image_bgr):
    # Normalize the bounding box coordinates
    normalized_bbox_coords = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        normalized_bbox_coords.append([
            x1 / image_bgr.shape[1],
            y1 / image_bgr.shape[0], 
            x2 / image_bgr.shape[1], 
            y2 / image_bgr.shape[0] 
        ])

    normalized_bbox_coords = torch.tensor(normalized_bbox_coords, dtype=torch.float32)
    return normalized_bbox_coords

def convert_patches_to_pixel_values(patches: List[Image.Image], patch_size: int = 32, patch_processor: CLIPImageProcessor = None):
    image_resized_patches = []
    
    for patch in patches:
        resized_patch = patch.resize((patch_size, patch_size), Image.LANCZOS) 
        inputs = patch_processor(images=resized_patch, return_tensors="pt")
        image_resized_patches.append(inputs['pixel_values'])
    
    return image_resized_patches

def create_batches(dataset, batch_size):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i:i + batch_size]

def pil_to_cv2(pil_image):
    open_cv_image = np.array(pil_image)
    if len(open_cv_image.shape) == 2:  # Grayscale image
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_GRAY2BGR)
    else:  # RGB image
        open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR
    return open_cv_image