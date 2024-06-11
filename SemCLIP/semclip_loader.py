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
    def __init__(self, config, text_pos_emb_2d=False):
        super().__init__(config)
        self.vision_model.embeddings.patch_embedding = nn.Linear(
            config.vision_config.patch_size * config.vision_config.patch_size * config.vision_config.num_channels,
            config.vision_config.hidden_size,
            bias=False
        )
        self.vision_model.embeddings.position_embedding = nn.Linear(4, config.vision_config.hidden_size, bias=False)
        
        if text_pos_emb_2d:
            self.text_model.embeddings.position_embedding = nn.Linear(4, config.text_config.hidden_size, bias=False)

    @classmethod
    def load_finetuned_model(cls, model_name, ignore_mismatched_sizes=True, text_pos_emb_2d=False, verbose=False):
        hf_model_name = model_mapper[model_name]
        config = CLIPConfig.from_pretrained(hf_model_name, token=HF_TOKEN)
        model = cls(config, text_pos_emb_2d)
        
        # Load state dict from the safetensors file        
        state_dict_path = hf_hub_download(repo_id=hf_model_name, filename="model.safetensors", use_auth_token=HF_TOKEN)
        state_dict = load_file(state_dict_path, device="cpu")
        print(f'device: {DEVICE}')

        # Extract and reshape weights for custom layers
        patch_embedding_weight = state_dict.pop('vision_model.embeddings.patch_embedding.weight')
        position_embedding_weight = state_dict.pop('vision_model.embeddings.position_embedding.weight')
        
        # Convert weights to float32 if they are in float16
        if patch_embedding_weight.dtype == torch.float16:
            patch_embedding_weight = patch_embedding_weight.to(torch.float32)
        if position_embedding_weight.dtype == torch.float16:
            position_embedding_weight = position_embedding_weight.to(torch.float32)
            
        # if 2D text positional embeddings are being used, extract and reshape the weights + convert to float32
        if text_pos_emb_2d:
            text_position_embedding_weight = state_dict.pop('text_model.embeddings.position_embedding.weight')
            if text_position_embedding_weight.dtype == torch.float16:
                text_position_embedding_weight = text_position_embedding_weight.to(torch.float32)

        model.load_state_dict(state_dict, strict=False)  # Load the remaining weights

        # Manually load custom layer weights
        model.vision_model.embeddings.patch_embedding.weight.data = patch_embedding_weight
        model.vision_model.embeddings.position_embedding.weight.data = position_embedding_weight
        if text_pos_emb_2d:
            model.text_model.embeddings.position_embedding.weight.data = text_position_embedding_weight
        
        if verbose:
            print(f'vision_model patch_embedding_shape: {patch_embedding_weight.shape}')
            print(f'vision_model patch_embedding_weight: {patch_embedding_weight}')
            print(f'vision_model position_embedding_shape: {position_embedding_weight.shape}')
            print(f'vision_model position_embedding_weight: {position_embedding_weight}')
            if text_pos_emb_2d:
                print(f'text_model position_embedding_shape: {model.text_model.embeddings.position_embedding.weight.shape}')
                print(f'text_model position_embedding_weight: {model.text_model.embeddings.position_embedding.weight}')

        return model
    