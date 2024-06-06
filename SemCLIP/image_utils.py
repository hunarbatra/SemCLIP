import cv2
import torch

import numpy as np

from transformers import CLIPImageProcessor
from datasets import Dataset
from PIL import Image
from typing import List


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def normalize_bbox_coords(bboxes, image_bgr):
    # Normalize the bounding box coordinates
    normalized_bbox_coords = []
    for bbox in bboxes:
        x1, y1, w, h = bbox
        normalized_bbox_coords.append([
            x1 / image_bgr.shape[1], # normalise x1 by width
            y1 / image_bgr.shape[0], # normalise y1 by height
            w / image_bgr.shape[1], # normalise segment width with image width
            h / image_bgr.shape[0] # normalise segment height with image height
        ])

    normalized_bbox_coords = torch.tensor(normalized_bbox_coords, dtype=torch.float32)
    return normalized_bbox_coords

def convert_patches_to_pixel_values(patches: List[Image.Image], patch_size: int = 32, patch_processor: CLIPImageProcessor = None):
    resized_patches = [patch.resize((patch_size, patch_size), Image.LANCZOS) for patch in patches]
    processed_patches = [patch_processor(images=resized_patch, return_tensors="pt")['pixel_values'] for resized_patch in resized_patches]
    
    return processed_patches

def create_batches(dataset, batch_size):
    dataset = dataset.shuffle(seed=42)
    
    for i in range(0, len(dataset), batch_size):
        yield dataset[i:i + batch_size]

def pil_to_cv2(pil_image):
    open_cv_image = np.array(pil_image)
    if len(open_cv_image.shape) == 2:  # Grayscale image
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_GRAY2BGR)
    else:  # RGB image
        open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR
    return open_cv_image