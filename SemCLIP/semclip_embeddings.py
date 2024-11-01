import torch
import torch.nn as nn

import numpy as np

from transformers.activations import ACT2FN

from SemCLIP.image_utils import DEVICE

from typing import List


class CLIPMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size).to(DEVICE)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size).to(DEVICE)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class SemCLIPMultiheadAttentionPoolingHead(nn.Module):
    """Multihead Attention Pooling for SemCLIP."""
    def __init__(self, config):
        super().__init__()
        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size)).to(DEVICE)
        self.attention = torch.nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True).to(DEVICE)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps).to(DEVICE)
        self.mlp = CLIPMLP(config)

    def forward(self, hidden_state):
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        hidden_state = self.attention(probe, hidden_state, hidden_state)[0]

        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)

        return hidden_state[:, 0]

class SemCLIPVisionEmbeddings(nn.Module):
    def __init__(self, vision_model, config, num_patches):
        super().__init__()
        self.model = vision_model
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_patches = num_patches
        self.patch_size = config.patch_size
        self.num_channels = config.num_channels
        self.num_patches = 50

        self.patch_dim = self.patch_size * self.patch_size * self.num_channels

        self.patch_norm1 = nn.LayerNorm(self.patch_dim).to(DEVICE)
        self.patch_norm2 = nn.LayerNorm(self.embed_dim).to(DEVICE)

        self.class_position_embedding = nn.Parameter(torch.randn(1, self.embed_dim), requires_grad=True).to(DEVICE) # Additional embedding for the class token that is prepended
        
        self.attn_pooling_head = SemCLIPMultiheadAttentionPoolingHead(config)
        
        # Register a buffer for positional IDs
        self.register_buffer("position_ids", torch.arange(self.num_patches).expand((1, -1)), persistent=False)
        
    def forward(self, patch_list: List, bbox_coords):
        patch_tensor = torch.stack(patch_list, dim=0).to(DEVICE) # [batch_size, 1, 3, 32, 32]
        batch_size = patch_tensor.size(0)
        
        # process individual patch embeddings just like ViTs original implementation 
        input_patch_tensor = patch_tensor.view(batch_size, self.num_channels, self.patch_size, self.patch_size) # [num_patches, 3, 32, 32]
        input_patch_tensor = input_patch_tensor.view(batch_size, -1) # flatten the patches - [num_patches, 3072]
        patch_norm = self.patch_norm1(input_patch_tensor) # apply layer norm (32*32*3) - shape: [num_patches, 3072]
        patch_embed = self.model.embeddings.patch_embedding(patch_norm) # pass the patch through a linear layer (32*32*3, 768) - shape: [num_patches, 768]
        patch_final = self.patch_norm2(patch_embed) # apply layer norm (768) - shape: [batch_size, 768]
        patch_embeds = patch_final.unsqueeze(0) # [1, num_patches, 768] -- 1 is the batch size here as currently we only process one image at a time, so we unsqueeze the batch dimension here

        # bbox positional embeddings
        bbox_coords = self.map_bbox_to_patch_idx(bbox_coords)
        sorted_patch_indices = torch.argsort(bbox_coords)
        
        # correctly ordered patch embeds based on their absolute position in the patch grid (1-49)
        reordered_patch_embeds = patch_embeds[:, sorted_patch_indices, :]
        
        # concatenate class patch embeddings and patch embeddings
        class_patch_embeds = self.model.embeddings.class_embedding.unsqueeze(0).unsqueeze(0) # [1, 1, 768]
        embeddings = torch.cat([class_patch_embeds, reordered_patch_embeds], dim=1)  # Concatenate class embeddings [1, 1, 768] to each patch embedding [1, batch_size, 768] (Concatenating along the patch dimension)
        
        if embeddings.shape[1] < self.num_patches:
            # it should be [1, 50, 768] so pad it if not
            pad_patch_embeds = torch.zeros((1, self.num_patches - embeddings.shape[1], embeddings.shape[2]), dtype=embeddings.dtype, device=embeddings.device)
            embeddings = torch.cat([embeddings, pad_patch_embeds], dim=1) # [1, 50, 768]

        # positional embeddings
        position_ids = self.position_ids # [1, 50]
        positional_embeddings = self.model.embeddings.position_embedding(position_ids) # [1, 50, 768]
        
        # add positional embeddings to patch embeddings
        patch_embeddings = embeddings[:, :self.num_patches, :] # cap the patch embeddings to the number of patches - shape: [1, 50, 768]
        
        embeddings = patch_embeddings + positional_embeddings # shape: [1, 50, 768]

        return embeddings
    
    def map_bbox_to_patch_idx(self, bbox_coords):
        patch_indices = []
        image_size = 224
        patch_size = 32
        num_patches = 7

        # Calculate the center coordinates
        c_x = (bbox_coords[:, 0] + bbox_coords[:, 2] / 2) * image_size
        c_y = (bbox_coords[:, 1] + bbox_coords[:, 3] / 2) * image_size
        
        # Determine the row and column in the patch grid
        row = torch.floor(c_y / patch_size).long()
        col = torch.floor(c_x / patch_size).long()
        
        # Ensure row and col are within bounds
        row = torch.clamp(row, 0, num_patches - 1)
        col = torch.clamp(col, 0, num_patches - 1)
        
        # Calculate the patch index (1-49)
        patch_indices = row * num_patches + col + 1
        
        return patch_indices.to(DEVICE)
    
    
class SemCLIPTextEmbeddings(nn.Module):
    def __init__(self, text_model, config, tokenizer):
        super().__init__()
        self.model = text_model
        self.config = config
        self.tokenizer = tokenizer
        self.embed_dim = config.hidden_size
        self.vocab_size = config.vocab_size
        self.max_position_embeddings = config.max_position_embeddings

        self.token_embedding = self.model.embeddings.token_embedding.to(DEVICE)
        
        self.attn_pooling_head = SemCLIPMultiheadAttentionPoolingHead(config)

    def forward(self, input_ids):
        seq_length = input_ids.shape[-1]
        
        token_embeddings = self.token_embedding(input_ids.to(DEVICE))

        token_positions = self.get_token_positions(input_ids).to(DEVICE)
        positional_embeddings = self.model.embeddings.position_embedding(token_positions)

        embeddings = token_embeddings + positional_embeddings

        return embeddings

    def get_token_positions(self, input_ids):
        batch_size, seq_length = input_ids.shape
        token_positions = []

        for batch in input_ids:
            token_pos = []
            start_idx = 0
            for token_id in batch:
                token = self.tokenizer.convert_ids_to_tokens(token_id.item())
                end_idx = start_idx + len(token)
                x = start_idx / seq_length
                y = start_idx / seq_length
                w = len(token) / seq_length
                h = len(token) / seq_length
                token_pos.append([x, y, w, h])
                start_idx = end_idx
            token_positions.append(token_pos)

        token_positions = torch.tensor(token_positions, dtype=torch.float32)
        return token_positions
        