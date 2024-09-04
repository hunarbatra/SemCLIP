dataset_mapper = {
    'COCO-13k': 'hunarbatra/CLIP-LLaVA-Instruct-COCO-13k',
    'imagenet1k-val': 'hunarbatra/imagenet1k_val_3k',
    # add more HF datasets later
}

dataset_keys = {
    'COCO-13k': {'img': 'downloaded_img', 'text': 'caption', 'img_idx': 'image'},
    'imagenet1k-val': {'img': 'image', 'text': 'label_name', 'img_idx': 'id'},
}

model_mapper = {
    'semclip-v1': 'hunarbatra/semclip-v1',
    'semclip-v1-wd-1e-3': 'hunarbatra/semclip-v1-wd-1e-3',
    'semclip-v1-epoch3': 'hunarbatra/semclip-v1-epoch3',
    'semclip-v4': 'hunarbatra/semclip-v4-1e4-64-text1Dposemb', # 1 epoch, lr=1e-4, text 1D positional embeddings
    'semclip-v4-cosine': 'hunarbatra/semclip-v4-1e3-64-cosine-text1Dposemb', # 1 epoch, lr=1e-3, cosine lr scheduler, text 1D positional embeddings
    'semclip-v4-cosine-2D': 'hunarbatra/semclip-v4-1e3-64-cosine', # 1 epoch, lr=1e-3, cosine lr scheduler, text 2D positional embeddings
    'clip': 'openai/clip-vit-base-patch32',
    'eva-clip': 'BAAI/EVA-CLIP-8B',
    'siglip': 'google/siglip-base-patch16-224',
    'dinov2': 'facebook/dinov2-small-imagenet1k-1-layer',
    'test-clip': 'hunarbatra/test-clip',
    'semclip-test': 'hunarbatra/test-clip',
    'semclip-v5-wip': 'hunarbatra/semclip-v5-wip',
    'semclip-v5': 'hunarbatra/semclip-v5',
    'clip-v1-test': 'hunarbatra/clip-v1-test',
    'semclip-v5-fixed': 'hunarbatra/semclip-v5-fixed',
    'semclip-v5-fixed-1e-4': 'hunarbatra/semclip-v5-fixed-1e-4',
}
