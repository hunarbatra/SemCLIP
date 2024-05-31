import torch
import torch.nn as nn
from transformers import CLIPTextConfig, CLIPTokenizer
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask

from .semclip_embeddings import SemCLIPTextEmbeddings
from .image_utils import DEVICE

from typing import Optional


class SemCLIPText(nn.Module):
    def __init__(self, model, text_config, tokenizer, pool_type='attention'):
        super(SemCLIPText, self).__init__()
        self.model = model
        self.text_config = text_config
        self.tokenizer = tokenizer
        self.pool_type = pool_type

    def forward(self, text):
        # Tokenize the input text -- Text Processor 
        inputs = self.tokenizer(text, return_tensors="pt").to(DEVICE)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Pass input_ids through custom SemCLIPTextEmbeddings module
        text_embeddings = SemCLIPTextEmbeddings(self.text_config, self.tokenizer).to(DEVICE)

        hidden_states = text_embeddings(input_ids)

        # CLIP's text model uses causal mask, prepare it here.
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
        last_hidden_state = self.model.text_model.final_layer_norm(last_hidden_state) # final layer norm is applied before pooling in huggingface transformers CLIP implentation

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
            raise ValueError(f"Invalid pooling type: {self.pool_type}")
        
        return pooled_output
