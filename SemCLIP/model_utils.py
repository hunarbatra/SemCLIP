import torch

import torch.nn as nn


def convert_models_to_fp32(model): 
    for name, p in model.named_parameters(): 
        p.data = p.data.float() 
        if p.grad is not None:
            p.grad.data = p.grad.data.float() # float32

def convert_models_to_fp16(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Conv1d):
            module.weight.data = module.weight.data.half()
            if module.bias is not None:
                module.bias.data = module.bias.data.half()
        elif isinstance(module, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(module, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()
        elif isinstance(module, torch.nn.BatchNorm2d):
            module.weight.data = module.weight.data.float()
            module.bias.data = module.bias.data.float()
        for name in ["text_projection", "proj"]:
            if hasattr(module, name):
                attr = getattr(module, name)
                if isinstance(attr, torch.Tensor):
                    attr.data = attr.data.half()
                    