import torch
import torch.nn as nn

from transformers.activations import ACT2FN

from SemCLIP.image_utils import DEVICE


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
    def __init__(self, config, num_patches):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_patches = num_patches
        self.patch_size = config.patch_size
        self.num_channels = config.num_channels

        self.class_embedding = nn.Parameter(torch.randn(1, 1, self.embed_dim), requires_grad=True).to(DEVICE)
        self.patch_dim = self.patch_size * self.patch_size * self.num_channels
        self.patch_embedding = nn.Linear(self.patch_dim, self.embed_dim, bias=False).to(DEVICE)

        self.patch_norm1 = nn.LayerNorm(self.patch_dim).to(DEVICE)
        self.patch_norm2 = nn.LayerNorm(self.embed_dim).to(DEVICE)

        self.position_embedding = nn.Linear(4, self.embed_dim).to(DEVICE)  # 4 for [x, y, w, h] - Positional embedding layer for bbox coordinates
        self.position_embedding.weight.requires_grad = True
        self.class_position_embedding = nn.Parameter(torch.randn(1, self.embed_dim), requires_grad=True).to(DEVICE) # Additional embedding for the class token that is prepended
        
        self.attn_pooling_head = SemCLIPMultiheadAttentionPoolingHead(config)

    def forward(self, patch_list, bbox_coords):
        # Ensure patch_list is on the correct device
        patch_list = [patch.to(self.patch_embedding.weight.device) for patch in patch_list]
        bbox_coords = bbox_coords.to(self.patch_embedding.weight.device)

        # process individual patch embeddings just like ViTs original implementation 
        patch_embeds_list = []
        for patch in patch_list: 
            patch_flat = patch.view(1, -1) # flatten the patch
            patch_norm = self.patch_norm1(patch_flat) # apply layer norm (32*32*3)
            patch_embed = self.patch_embedding(patch_norm) # pass the patch through a linear layer (32*32*3, 768)
            patch_final = self.patch_norm2(patch_embed) # apply layer norm (768)
            patch_embeds_list.append(patch_embed)

        # patch embeddings
        patch_embeds = torch.stack(patch_embeds_list, dim=1)  # stack along dimension 1
        embeddings = torch.cat([self.class_embedding, patch_embeds], dim=1)  # Concatenate class embeddings to each patch embedding (Concatenating along the patch dimension)

        # bbox positional embeddings
        positional_embeddings = self.position_embedding(bbox_coords).unsqueeze(0)
        class_pos_embedding = self.class_position_embedding.unsqueeze(0)
        positional_embeddings = torch.cat([class_pos_embedding, positional_embeddings], dim=1)

        embeddings += positional_embeddings

        return embeddings
    
    
class SemCLIPTextEmbeddings(nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.embed_dim = config.hidden_size
        self.vocab_size = config.vocab_size
        self.max_position_embeddings = config.max_position_embeddings

        self.token_embedding = nn.Embedding(self.vocab_size, self.embed_dim).to(DEVICE)
        self.token_embedding.weight.requires_grad = True
        self.position_embedding = nn.Linear(4, self.embed_dim).to(DEVICE)  # 4 for [x, y, w, h] - Positional embedding layer for token coordinates
        self.position_embedding.weight.requires_grad = True
        
        self.attn_pooling_head = SemCLIPMultiheadAttentionPoolingHead(config)

    def forward(self, input_ids):
        seq_length = input_ids.shape[-1]
        
        token_embeddings = self.token_embedding(input_ids.to(DEVICE))

        token_positions = self.get_token_positions(input_ids).to(DEVICE)
        positional_embeddings = self.position_embedding(token_positions)

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
        