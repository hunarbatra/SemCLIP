import torch
import argparse

import torch.nn as nn

from transformers import CLIPProcessor, CLIPModel, CLIPVisionConfig, CLIPImageProcessor, CLIPTextConfig, CLIPTokenizer
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask

from PIL import Image

from .semclip_tokenization import preprocess_patches
from .semclip_embeddings import SemCLIPVisionEmbeddings, SemCLIPTextEmbeddings
from .image_utils import convert_patches_to_pixel_values, DEVICE

from typing import Optional


# Load the CLIP model, processor and model config
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
patch_processor = CLIPImageProcessor(do_resize=False, do_center_crop=False) # disable image resizing and center cropping

vision_config = CLIPVisionConfig.from_pretrained("openai/clip-vit-base-patch32")
text_config = CLIPTextConfig.from_pretrained("openai/clip-vit-base-patch32")

model.to(DEVICE)

def get_segments_embeddings(image_name: str, data_name: str, projection_dim: Optional[int] = None):
    # Load image segments patches
    image_resized_patches, normalized_bbox_coords = preprocess_patches(image_name, data_name)
    image_resized_patches = convert_patches_to_pixel_values(image_resized_patches, patch_size=vision_config.patch_size, patch_processor=patch_processor)
    
    if projection_dim is not None: # custom projection dim, default is 512
        vision_config.projection_dim = projection_dim

    # Pass image patches through custom CLIPVisionEmbeddings module
    patch_embeddings = SemCLIPVisionEmbeddings(vision_config, len(image_resized_patches))
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
    
    # Detach the final embedding
    final_embedding = final_embedding.detach()
    
    return final_embedding

def get_text_embeddings(text: str, projection_dim: Optional[int] = None):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    if projection_dim is not None:  # custom projection dim, default is 512
        text_config.projection_dim = projection_dim

    # Pass input_ids through custom SemCLIPTextEmbeddings module
    text_embeddings = SemCLIPTextEmbeddings(text_config, tokenizer)
    with torch.no_grad():
        hidden_states = text_embeddings(input_ids)

    # CLIP's text model uses causal mask, prepare it here.
    # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
    input_shape = input_ids.size()
    causal_attention_mask = _create_4d_causal_attention_mask(
        input_shape, hidden_states.dtype, device=hidden_states.device
    )
    # expand attention_mask
    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

    # Pass the hidden states through the encoder
    encoder_outputs = model.text_model.encoder(
        inputs_embeds=hidden_states,
        attention_mask=attention_mask,
        causal_attention_mask=causal_attention_mask,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )

    # Get the last hidden state and apply final layer normalization
    last_hidden_state = encoder_outputs.last_hidden_state
    last_hidden_state = model.text_model.final_layer_norm(last_hidden_state)

    if model.text_model.config.eos_token_id == 2:
        # The `eos_token_id` was incorrect before PR #24773: Let's keep what have been done here.
        # A CLIP model with such `eos_token_id` in the config can't work correctly with extra new tokens added
        # ------------------------------------------------------------
        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
        ]
    else:
        # The config gets updated `eos_token_id` from PR #24773 (so the use of exta new tokens is possible)
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            # We need to get the first position of `eos_token_id` value (`pad_token_ids` might equal to `eos_token_id`)
            # Note: we assume each sequence (along batch dim.) contains an  `eos_token_id` (e.g. prepared by the tokenizer)
            (input_ids.to(dtype=torch.int, device=last_hidden_state.device) == model.text_model.config.eos_token_id)
            .int()
            .argmax(dim=-1),
        ]

    # Pass the pooled output through a final linear projection
    text_projection = nn.Linear(text_config.hidden_size, text_config.projection_dim, bias=False)
    final_embedding = text_projection(pooled_output)

    # Detach the final embedding
    final_embedding = final_embedding.detach()

    return final_embedding


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_name", type=str, help="image file name")
    parser.add_argument("--data_name", type=str, help="data directory name")
    parser.add_argument("--projection_dim", type=int, default=None, help="custom projection dimension, default: 512")
    args = parser.parse_args()
    get_segments_embeddings(args.image_name, args.data_name, args.projection_dim)