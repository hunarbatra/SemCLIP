import torch
import cv2
import fire

import supervision as sv
import matplotlib.pyplot as plt
import numpy as np

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from PIL import Image

from .image_utils import normalize_bbox_coords


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"

sam = sam_model_registry[MODEL_TYPE](checkpoint='./sam-weights/sam_vit_h_4b8939.pth').to(device=DEVICE)
sam = sam.to(device=DEVICE)

def save_original_and_crops(original_image_path, cropped_images, segmented_image, save_path):
    """
    Plots the original image and all cropped images with transparent backgrounds in a grid.

    Args:
    - original_image_path (str): Path to the original image.
    - cropped_images (List[np.ndarray]): List of cropped images with transparent backgrounds as numpy arrays.
    """
    # Load and convert the original image to RGB
    original_image = cv2.imread(original_image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    segmented_image_rgb = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)

    # Determine grid size for plotting
    n_crops = len(cropped_images)
    n_cols = 3  # Number of columns in the grid
    n_rows = 2 + (n_crops // n_cols)  # two extra row for the original image and segmented image

    plt.figure(figsize=(n_cols * 4, n_rows * 4))

    # Plot the original image
    plt.subplot(n_rows, n_cols, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(n_rows, n_cols, 2)
    plt.imshow(segmented_image_rgb)
    plt.title("Segmented Image")
    plt.axis("off")

    # Plot each crop
    for i, crop in enumerate(cropped_images, start=3):
        plt.subplot(n_rows, n_cols, i)
        # If the crop has an alpha channel, convert it to RGBA for matplotlib
        if crop.shape[2] == 4:  # Assuming the crop is in BGRA format
            crop_rgba = cv2.cvtColor(crop, cv2.COLOR_BGRA2RGBA)
            plt.imshow(crop_rgba)
        else:  # For non-transparent (RGB) crops, if any
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            plt.imshow(crop_rgb)
        plt.title(f"Crop {i-2}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def generate_crops_from_detections(image_path, detections, annotated_image, save_crops=False):
    """
    Generates square crops from detections with transparent backgrounds and saves them.

    Args:
    - image_path (str): Path to the original image.
    - detections (Detections): Detections object containing all the segments.

    Returns:
    - List of numpy arrays of the cropped images.
    """
    save_data_name = image_path.split('/')[-2]
    save_img_name = f"{image_path.split('/')[-1].split('.')[0]}_crops.png"
    save_path = f"data/{save_data_name}/{save_img_name}"

    # Load the original image in BGRA mode (to handle the alpha channel)
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    h, w = image.shape[:2]

    if image.shape[2] == 3:  # Convert BGR to BGRA if necessary
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    cropped_images = []

    for i, (bbox, mask) in enumerate(zip(detections.xyxy, detections.mask)):
        x1, y1, x2, y2 = bbox
        crop_width = x2 - x1
        crop_height = y2 - y1

        # Crop the segment
        segment_crop = image[y1:y2, x1:x2]

        # Determine the size to pad to make the crop square
        size_difference = abs(crop_width - crop_height)
        padding_top = padding_bottom = padding_left = padding_right = 0

        if crop_width > crop_height:
            padding_top = padding_bottom = size_difference // 2
            # Adjust for odd difference
            if size_difference % 2 != 0:
                padding_bottom += 1
        else:
            padding_left = padding_right = size_difference // 2
            # Adjust for odd difference
            if size_difference % 2 != 0:
                padding_right += 1

        # Create an alpha channel for the segment crop
        alpha_channel = np.zeros((crop_height, crop_width), dtype=np.uint8)
        alpha_channel[mask[y1:y2, x1:x2]] = 255

        # Combine the segment crop with the alpha channel
        segment_crop_bgra = cv2.cvtColor(segment_crop, cv2.COLOR_BGR2BGRA)
        segment_crop_bgra[:, :, 3] = alpha_channel  # Set the alpha channel

        # Pad the crop to make it square
        square_crop = cv2.copyMakeBorder(segment_crop_bgra, padding_top, padding_bottom, padding_left, padding_right, cv2.BORDER_CONSTANT, value=[0, 0, 0, 0])

        cropped_images.append(square_crop)
        
    if save_crops:
        save_original_and_crops(image_path, cropped_images, annotated_image, save_path)

    return cropped_images

def preprocess_patches(image_name: str, data_name: str, save_crops: bool = False):
    image_path = f"data/{data_name}/{image_name}"
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # SAM automated mask generation
    mask_generator = SamAutomaticMaskGenerator(sam)
    sam_result = mask_generator.generate(image_rgb)
    
    # Extract bounding boxes coordinates for positional embeddings from the SAM result
    bboxes = [mask['bbox'] for mask in sam_result]
    normalized_bbox_coords = normalize_bbox_coords(bboxes, image_bgr) # tensor
    
    # Annotate the image with the detections
    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
    detections = sv.Detections.from_sam(sam_result=sam_result)
    annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)
    
    # Generate crops from detections and save them
    cropped_images_detection = generate_crops_from_detections(image_path, detections, annotated_image, save_crops)
    cropped_images_detection = [Image.fromarray(crop) for crop in cropped_images_detection]
    
    return cropped_images_detection, normalized_bbox_coords

    
if __name__ == "__main__":
    fire.Fire(preprocess_patches)
    