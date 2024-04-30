import torch
import fire

import torch.nn as nn

from transformers import CLIPProcessor, CLIPModel, CLIPVisionConfig, CLIPImageProcessor
from PIL import Image

from sam_patches import preprocess_patches
from patch_embeddings import CLIPVisionEmbeddings
from image_utils import convert_patches_to_pixel_values


# Load the CLIP model, processor and model config
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
patch_processor = CLIPImageProcessor(do_resize=False, do_center_crop=False) # disable image resizing and center cropping
vision_config = CLIPVisionConfig.from_pretrained("openai/clip-vit-base-patch32")

def get_segments_embeddings(image_name: str, data_name: str):
    # Load image segments patches
    image_resized_patches, normalized_bbox_coords = preprocess_patches(image_name, data_name)
    image_resized_patches = convert_patches_to_pixel_values(image_resized_patches, patch_size=vision_config.patch_size, patch_processor=patch_processor)

    # Pass image patches through custom CLIPVisionEmbeddings module
    patch_embeddings = CLIPVisionEmbeddings(vision_config, len(image_resized_patches))
    with torch.no_grad():
        output_embeddings = patch_embeddings(image_resized_patches, normalized_bbox_coords)
        
    # Apply pre-layer normalization
    hidden_states = model.vision_model.pre_layrnorm(output_embeddings)

    # Pass the normalized embeddings through the encoder
    encoder_outputs = model.vision_model.encoder(
        inputs_embeds=hidden_states,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )

    # Get the last hidden state and apply post-layer normalization
    last_hidden_state = encoder_outputs.last_hidden_state
    pooled_output = last_hidden_state[:, 0, :]
    pooled_output = model.vision_model.post_layernorm(pooled_output)

    # Pass the pooled output through a final linear projection
    visual_projection = nn.Linear(vision_config.hidden_size, vision_config.projection_dim, bias=False)
    final_embedding = visual_projection(pooled_output)
    
    print(final_embedding.shape)
    
    print(final_embedding)

if __name__ == "__main__":
    fire.Fire(get_segments_embeddings)