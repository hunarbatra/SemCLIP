dataset_mapper = {
    'COCO-13k': 'hunarbatra/CLIP-LLaVA-Instruct-COCO-13k',
    # add more HF datasets later
}

model_mapper = {
    'semclip-v1': 'hunarbatra/semclip-v1',
    'semclip-v1-wd-1e-3': 'hunarbatra/semclip-v1-wd-1e-3',
    'semclip-v1-epoch3': 'hunarbatra/semclip-v1-epoch3',
    'semclip-v4': 'hunarbatra/semclip-v4-1e4-64-text1Dposemb', # 1 epoch, lr=1e-4, text 1D positional embeddings
    'semclip-v4-cosine': 'hunarbatra/semclip-v4-1e3-64-cosine-text1Dposemb', # 1 epoch, lr=1e-3, cosine lr scheduler, text 1D positional embeddings
    'semclip-v4-cosine-2D': 'hunarbatra/semclip-v4-1e3-64-cosine', # 1 epoch, lr=1e-3, cosine lr scheduler, text 2D positional embeddings
    'clip': 'openai/clip-vit-base-patch32',
    'eva-clip': 'QuanSun/EVA-CLIP',
    'siglip': 'google/siglip-base-patch16-224',
    'dinov2': 'facebook/dinov2-base',
}
