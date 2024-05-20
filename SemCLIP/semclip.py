import os

import torch
import argparse

import torch.nn as nn
import numpy as np

from transformers import CLIPProcessor, CLIPModel, CLIPVisionConfig, CLIPImageProcessor, CLIPTextConfig, CLIPTokenizer
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask

from huggingface_hub import HfApi, Repository

from PIL import Image

from .semclip_tokenization import preprocess_patches
from .semclip_embeddings import SemCLIPVisionEmbeddings, SemCLIPTextEmbeddings
from .image_utils import convert_patches_to_pixel_values, DEVICE

from typing import Optional
from dotenv import load_dotenv


load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


class SemCLIP:
    def __init__(self, model_name="openai/clip-vit-base-patch32", pool_type: str = 'attention', projection_dim: Optional[int] = None, device=DEVICE):
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name, token=HF_TOKEN).to(device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.patch_processor = CLIPImageProcessor(do_resize=False, do_center_crop=False)  # disable image resizing and center cropping
        self.vision_config = CLIPVisionConfig.from_pretrained(model_name)
        self.text_config = CLIPTextConfig.from_pretrained(model_name)
        self.pool_type = pool_type
        
        if projection_dim is not None:  # custom projection dim, default is 512
            self.vision_config.projection_dim = projection_dim
            
        # Projection layers
        self.visual_projection = nn.Linear(self.vision_config.hidden_size, self.vision_config.projection_dim, bias=False).to(self.device)
        self.text_projection = nn.Linear(self.text_config.hidden_size, self.text_config.projection_dim, bias=False).to(self.device)
        self.logit_scale = nn.Parameter(torch.tensor(self.model.config.logit_scale_init_value)).to(self.device)
        
    def get_segment_embeddings(self, image_name: str, data_name: str, image_file: Optional[np.ndarray] = None):
        # Load image segments patches
        image_resized_patches, normalized_bbox_coords = preprocess_patches(image_name, data_name, image_file=image_file)
        image_resized_patches = convert_patches_to_pixel_values(image_resized_patches, patch_size=self.vision_config.patch_size, patch_processor=self.patch_processor)
            
        # Pass image patches through custom CLIPVisionEmbeddings module
        patch_embeddings = SemCLIPVisionEmbeddings(self.vision_config, len(image_resized_patches)).to(self.device)
        with torch.no_grad():
            output_embeddings = patch_embeddings(image_resized_patches, normalized_bbox_coords)
            
        # Apply pre-layer normalization
        # additional layer normalization to the combined patch & post embeddings before the transformer (clip)
        hidden_states = self.model.vision_model.pre_layrnorm(output_embeddings)
        
        # Pass the normalized embeddings through the CLIP transformer encoder
        encoder_outputs = self.model.vision_model.encoder(
            inputs_embeds=hidden_states,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        
        # Get the last hidden state, apply pooling and apply post-layer normalization
        last_hidden_state = encoder_outputs.last_hidden_state
        
        if self.pool_type == 'cls':
            pooled_output = last_hidden_state[:, 0, :]
        elif self.pool_type == 'mean':
            pooled_output = last_hidden_state.mean(dim=1)
        elif self.pool_type == 'attention':
            pooled_output = patch_embeddings.attn_pooling_head(last_hidden_state)
        else:
            raise ValueError(f"Invalid pooling type: {pool_type}")
        
        pooled_output = self.model.vision_model.post_layernorm(pooled_output)  # post layer norm
        
        # Pass the pooled output through a final linear projection
        final_embedding = self.visual_projection(pooled_output)
        
        # Detach the final embedding
        final_embedding = final_embedding.detach()
        
        return final_embedding
    
    def get_text_embeddings(self, text: str):
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
            
        # Pass input_ids through custom SemCLIPTextEmbeddings module
        text_embeddings = SemCLIPTextEmbeddings(self.text_config, self.tokenizer).to(self.device)
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
        encoder_outputs = self.model.text_model.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        
        # Get the last hidden state, apply pooling and apply final layer normalization
        last_hidden_state = encoder_outputs.last_hidden_state
        
        if self.pool_type == 'cls':
            if self.model.text_model.config.eos_token_id == 2:
                # The `eos_token_id` was incorrect before PR #24773: Let's keep what have been done here.
                # A CLIP model with such `eos_token_id` in the config can't work correctly with extra new tokens added
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
                    (input_ids.to(dtype=torch.int, device=last_hidden_state.device) == self.model.text_model.config.eos_token_id)
                    .int()
                    .argmax(dim=-1),
                ]
        elif self.pool_type == 'mean':
            pooled_output = last_hidden_state.mean(dim=1)
        elif self.pool_type == 'attention':
            pooled_output = text_embeddings.attn_pooling_head(last_hidden_state)
        else:
            raise ValueError(f"Invalid pooling type: {pool_type}")
        
        pooled_output = self.model.text_model.final_layer_norm(pooled_output)
        
        # Pass the pooled output through a final linear projection
        final_embedding = self.text_projection(pooled_output)
        
        # Detach the final embedding
        final_embedding = final_embedding.detach()
        
        return final_embedding
    
    def get_semclip_embeddings(self, images, captions, images_folder):
        image_embeddings = []
        text_embeddings = []

        for image, caption in zip(images, captions):
            # Assuming get_segment_embeddings and get_text_embeddings expect a single input
            image_embedding = self.get_segment_embeddings(image_name=image, data_name=images_folder)
            text_embedding = self.get_text_embeddings(text=caption)

            image_embeddings.append(image_embedding)
            text_embeddings.append(text_embedding)

        image_embeddings = torch.cat(image_embeddings, dim=0)
        text_embeddings = torch.cat(text_embeddings, dim=0)

        return image_embeddings, text_embeddings
    
    def get_semclip_embeddings_direct_img(self, images, captions):
        image_embeddings = []
        text_embeddings = []

        ctr = 0

        for image, caption in zip(images, captions):
            print(ctr)
            ctr += 1
            # Assuming get_segment_embeddings and get_text_embeddings expect a single input
            image_embedding = self.get_segment_embeddings(image_name=None, data_name=None, image_file=image)
            text_embedding = self.get_text_embeddings(text=caption)

            image_embeddings.append(image_embedding)
            text_embeddings.append(text_embedding)

        image_embeddings = torch.cat(image_embeddings, dim=0)
        text_embeddings = torch.cat(text_embeddings, dim=0)

        return image_embeddings, text_embeddings
    
    def process_final_embeddings(self, image_embeddings, text_embeddings):
        # Apply L2 normalization
        image_embeddings = image_embeddings / image_embeddings.norm(p=2, dim=-1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(p=2, dim=-1, keepdim=True)

        # Compute cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeddings, image_embeddings.t()) * logit_scale
        logits_per_image = logits_per_text.t()

        return logits_per_image, logits_per_text
    
    def upload_model_to_hf_hub(self, model_name: str, hf_name: str):
        api = HfApi()

        repo_exists = api.repo_exists(repo_id=f"{hf_name}/{model_name}", token=HF_TOKEN)
        
        print(f"Repository exists: {repo_exists}")

        if repo_exists:
            # Clone the existing repository to a local directory
            repo = Repository(local_dir=model_name, clone_from=f"https://huggingface.co/{hf_name}/{model_name}", token=HF_TOKEN)
            commit_message = "Update model files"
        else:
            # Create a new repository
            repo_url = api.create_repo(repo_id=model_name, token=HF_TOKEN, private=True)
            # Clone the new repository to a local directory
            repo = Repository(local_dir=model_name, clone_from=repo_url, use_auth_token=HF_TOKEN)
            commit_message = "Add model files"

        self.model.save_pretrained(model_name)
        self.tokenizer.save_pretrained(model_name)
        self.processor.save_pretrained(model_name)
        
        # Push the files to the repository
        repo.push_to_hub(commit_message=commit_message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_name", type=str, help="image file name")
    parser.add_argument("--data_name", type=str, help="data directory name")
    parser.add_argument("--pool_type", type=str, default="mean", help="pooling type; options: ['mean', 'cls', 'attention'], default: 'attention'")
    parser.add_argument("--projection_dim", type=int, default=None, help="custom projection dimension, default: 512")
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32", help="CLIP model name, default: openai/clip-vit-base-patch32")
    args = parser.parse_args()
    
    semclip = SemCLIP(model_name=args.model_name, pool_type=args.pool_type, projection_dim=args.projection_dim, device=DEVICE)
    semclip.get_segment_embeddings(args.image_name, args.data_name)
    