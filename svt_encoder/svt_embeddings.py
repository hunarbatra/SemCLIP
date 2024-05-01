import torch
import torch.nn as nn

class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, config, num_patches):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_patches = num_patches
        self.patch_size = config.patch_size
        self.num_channels = config.num_channels

        self.class_embedding = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.patch_dim = self.patch_size * self.patch_size * self.num_channels
        self.patch_embedding = nn.Linear(self.patch_dim, self.embed_dim, bias=False)

        self.patch_norm1 = nn.LayerNorm(self.patch_dim)
        self.patch_norm2 = nn.LayerNorm(self.embed_dim)

        self.position_embedding = nn.Linear(4, self.embed_dim)  # 4 for [x, y, w, h] - Positional embedding layer for bbox coordinates
        self.class_position_embedding = nn.Parameter(torch.randn(1, self.embed_dim)) # Additional embedding for the class token that is prepended


    def forward(self, patch_list, bbox_coords):
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
