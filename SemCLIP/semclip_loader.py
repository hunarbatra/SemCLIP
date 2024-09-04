import os

import torch
import torch.nn as nn

from transformers import CLIPModel, CLIPConfig
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from dotenv import load_dotenv

from SemCLIP.image_utils import DEVICE
from config import model_mapper


load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

class SemCLIPLoader(CLIPModel):
    def __init__(self, config):
        super().__init__(config)
        self.vision_model.embeddings.patch_embedding = nn.Linear(
            config.vision_config.patch_size * config.vision_config.patch_size * config.vision_config.num_channels,
            config.vision_config.hidden_size,
            bias=False
        ) # [768, 3072]
        
        # self.vision_model.embeddings.position_embedding = nn.Linear(4, config.vision_config.hidden_size, bias=False)
        
    @classmethod
    def init_model_for_finetuning(cls, model_name, verbose=False):
        config = CLIPConfig.from_pretrained(model_name, token=HF_TOKEN)
        model = CLIPModel.from_pretrained(
            model_name, 
            token=HF_TOKEN,
            ignore_mismatched_sizes=True, # needed when finetuning a finetuned CLIP modified model
        ).to(DEVICE)
        
        original_clip_patch_embedding_weights = model.vision_model.embeddings.patch_embedding.weight.data # [768, 3, 32, 32]
        clip_patch_embedding_weights = original_clip_patch_embedding_weights.reshape(config.vision_config.hidden_size, -1) # [768, 3072]
        
        # modify the archicture of patch_embedding layer to take in [768, 3072] instead of [768, 3, 32, 32]
        model.vision_model.embeddings.patch_embedding = nn.Linear(
            config.vision_config.patch_size * config.vision_config.patch_size * config.vision_config.num_channels,
            config.vision_config.hidden_size,
            bias=False
        ).to(DEVICE)
        
        if clip_patch_embedding_weights.dtype == torch.float16:
            clip_patch_embedding_weights = clip_patch_embedding_weights.to(torch.float32)
            
        model.vision_model.embeddings.patch_embedding.weight.data = clip_patch_embedding_weights
        
        if verbose:
            print(f'original CLIP patch_embedding shape: {original_clip_patch_embedding_weights.shape}')
            print(f'updated CLIP patch_embedding shape: {model.vision_model.embeddings.patch_embedding.weight.shape}')
            print(f'original CLIP patch_embedding data: {original_clip_patch_embedding_weights}')
            print(f'updated CLIP patch_embedding data: {model.vision_model.embeddings.patch_embedding.weight.data}')
        
        return model

    @classmethod
    def load_finetuned_model(cls, model_name, verbose=False):
        config = CLIPConfig.from_pretrained(model_name, token=HF_TOKEN)
        model = cls(config)
        
        # Load state dict from the safetensors file        
        state_dict_path = hf_hub_download(repo_id=model_name, filename="model.safetensors", use_auth_token=HF_TOKEN)
        state_dict = load_file(state_dict_path, device="cpu")

        # Extract and reshape weights for custom layers
        patch_embedding_weight = state_dict.pop('vision_model.embeddings.patch_embedding.weight')
        
        # Convert weights to float32 if they are in float16
        if patch_embedding_weight.dtype == torch.float16:
            patch_embedding_weight = patch_embedding_weight.to(torch.float32)
    
        model.load_state_dict(state_dict, strict=False)  # Load the remaining weights

        # Manually load custom layer weights
        model.vision_model.embeddings.patch_embedding.weight.data = patch_embedding_weight

        if verbose:
            print(f'vision_model patch_embedding_shape: {patch_embedding_weight.shape}')
            print(f'vision_model patch_embedding_weight: {patch_embedding_weight}')

        return model
    