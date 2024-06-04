import torch
import torch.nn as nn
from transformers import CLIPVisionConfig

from .semclip_embeddings import SemCLIPVisionEmbeddings
from .image_utils import DEVICE


class SemCLIPVision():
    def __init__(self, vision_model, vision_config, pool_type='attention'):
        self.vision_model = vision_model
        self.vision_config = vision_config
        self.pool_type = pool_type
        
    def forward(self, image_resized_patches, normalized_boox_coords):
        # Pass image patches through custom CLIPVisionEmbeddings module
        patch_embeddings = SemCLIPVisionEmbeddings(self.vision_model, self.vision_config, len(image_resized_patches)).to(DEVICE)
        output_embeddings = patch_embeddings(image_resized_patches, normalized_boox_coords)
        
        # apply pre-layer normalization
        # additional layer norm to the combined patch + positional embeddings before passing onto the transformer encoder
        hidden_states = self.vision_model.pre_layrnorm(output_embeddings)
        
        # pass the normalized embeddings through the CLIP transformer encoder
        encoder_outputs = self.vision_model.encoder(
            inputs_embeds=hidden_states,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        
        # get the last hidden state from the transformer encoder
        last_hidden_state = encoder_outputs.last_hidden_state # equivalent to doing encoder_outputs[0] as in the huggingface transformers CLIP code
        
        # apply pooling
        if self.pool_type == 'cls':
            pooled_output = last_hidden_state[:, 0, :]
        elif self.pool_type == 'mean':
            pooled_output = last_hidden_state.mean(dim=1)
        elif self.pool_type == 'attention':
            pooled_output = patch_embeddings.attn_pooling_head(last_hidden_state)
        else:
            raise ValueError(f"Invalid pooling type: {self.pool_type}")
        
        # post layer normalization
        pooled_output = self.vision_model.post_layernorm(pooled_output)
        
        return pooled_output
