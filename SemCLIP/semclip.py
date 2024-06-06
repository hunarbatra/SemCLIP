import os
import torch
import argparse
import torch.nn as nn
import numpy as np

from transformers import CLIPProcessor, CLIPModel, CLIPVisionConfig, CLIPTextConfig, CLIPTokenizer, CLIPImageProcessor
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import HfApi, Repository
from dotenv import load_dotenv
from typing import Optional 

from .semclip_tokenization import preprocess_patches
from .semclip_vision import SemCLIPVision
from .semclip_text import SemCLIPText
from .image_utils import convert_patches_to_pixel_values, DEVICE


load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


class SemCLIPImageFeatures():
    def __init__(self, vision_config, vision_model, visual_projection, pool_type, device=DEVICE):
        self.device = device
        self.vision_model = vision_model
        self.visual_projection = visual_projection

    def forward(self, image_patches, bbox_coords):
        # SemCLIP vision model forward pass which returns the pooled output
        pooled_output = self.vision_model.forward(image_patches, bbox_coords)
        # pass the pooled output through a final linear projection layer
        image_embeds = self.visual_projection(pooled_output)
        return image_embeds

class SemCLIPTextFeatures():
    def __init__(self, text_config, tokenizer, text_model, text_projection, pool_type, text_pos_emb_2d=True, device=DEVICE):
        self.device = device
        
        self.text_model = text_model
        self.text_projection = text_projection
        
        self.text_pos_emb_2d = text_pos_emb_2d


    def forward(self, text):
        if self.text_pos_emb_2d:
            # SemCLIP text model forward pass which returns the pooled output
            # applies text processor, generates SemCLIPTextEmbeddings with 2D positional embeddings, and passes through the CLIP transformer encoder followed by pooling 
            pooled_output = self.text_model.forward(text)
        else:
            pooled_output = self.text_model(**text).pooler_output # HuggingFace text model with 1D positional embeddings
            
        # pass the pooled output through a final linear projection layer
        text_embeds = self.text_projection(pooled_output)
        
        return text_embeds

class SemCLIP(nn.Module):
    def __init__(
        self, 
        model_name="openai/clip-vit-base-patch32", 
        pool_type='attention', 
        projection_dim=None, 
        ignore_mismatched_sizes=False, 
        text_pos_emb_2d=True,
        device=DEVICE,
    ):
        super(SemCLIP, self).__init__()
        self.device = device
        self.ignored_mismatched_sizes = ignore_mismatched_sizes
        self.text_pos_emb_2d = text_pos_emb_2d
        
        self.model = CLIPModel.from_pretrained(
            model_name, 
            token=HF_TOKEN, 
            ignore_mismatched_sizes=self.ignored_mismatched_sizes
        ).to(device)
        
        # Store original state dict to preserve weight loading for custom layers
        # self.original_state_dict = self.model.state_dict()
        
        # Modify specific layers
        self.model.vision_model.embeddings.patch_embedding = nn.Linear(
            self.model.vision_model.config.patch_size * self.model.vision_model.config.patch_size * self.model.vision_model.config.num_channels, 
            self.model.vision_model.config.hidden_size, 
            bias=False
        ).to(device) # updated patch embedding layer
        self.model.vision_model.embeddings.position_embedding = nn.Linear(4, self.model.vision_model.config.hidden_size, bias=False).to(device) # updated position embedding layer for vision model with 2D positional embeddings -- 4 for [x, y, w, h] - Positional embedding layer for bbox coordinates
        if self.text_pos_emb_2d: # 2D positional embeddings for text
            self.model.text_model.embeddings.position_embedding = nn.Linear(4, self.model.text_model.config.hidden_size, bias=False).to(device) # updated position embedding layer for text model with 2D positional embeddings -- 4 for [x, y, w, h] - Positional embedding layer for token coordinates

        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.patch_processor = CLIPImageProcessor(do_resize=False, do_center_crop=False)  # disable image resizing and center cropping
        self.pool_type = pool_type

        if projection_dim is not None:
            self.model.vision_model.config.projection_dim = projection_dim
            self.model.text_model.config.projection_dim = projection_dim

        self.vision_model = SemCLIPVision(self.model.vision_model, self.model.vision_model.config, self.pool_type)
        self.text_model = SemCLIPText(self.model.text_model, self.model.text_model.config, self.tokenizer, self.pool_type) if self.text_pos_emb_2d else self.model.text_model # use 2D positional embeddings for text or original 1D positional embeddings (default: True)

        self.vision_features_extractor = SemCLIPImageFeatures(self.model.vision_model.config, self.vision_model, self.model.visual_projection, self.pool_type)
        self.text_features_extractor = SemCLIPTextFeatures(self.model.text_model.config, self.tokenizer, self.text_model, self.model.text_projection, self.pool_type, self.text_pos_emb_2d)
        
        print(f'Using 2D position embeddings for text_model: {self.text_pos_emb_2d}')

    def get_image_features(self, image_name: str, data_name: str, image_file: Optional[np.ndarray] = None, return_embeds=False):
        # load image segments patches and normalized bounding box coordinates
        image_resized_patches, normalized_bbox_coords = preprocess_patches(image_name, data_name, image_file=image_file)
        # preprocess the patches (CLIPImageProcessor)
        image_resized_patches = convert_patches_to_pixel_values(image_resized_patches, patch_size=self.model.vision_model.config.patch_size, patch_processor=self.patch_processor) # list of torch.tensor
        
        normalized_bbox_coords = normalized_bbox_coords.to(self.device)

        if not return_embeds:
            return image_resized_patches, normalized_bbox_coords

        image_features = self.vision_features_extractor.forward(image_resized_patches, normalized_bbox_coords)

        return image_features

    def get_text_features(self, text, return_embeds=False): 
        if not self.text_pos_emb_2d:
            # preprocess text if 1D positional embeddings are used
            # SemCLIP 2D positional embeddings internally tokenizes the input text and passes it through the CLIP transformer encoder followed by pooling, so it's not needed there
            text = self.tokenizer(text, padding=True, return_tensors="pt").to(DEVICE)
            
        if not return_embeds:
            return text

        text_features = self.text_features_extractor.forward(text)

        return text_features
    
    def generate_image_text_embeddings(self, image, caption, image_folder):
        if image_folder is None:
            image_file = image
            image_name = None
        else:
            image_file = None
            image_name = image
            
        image_patches, bbox_coords = self.get_image_features(image_name=image_name, data_name=image_folder, image_file=image_file, return_embeds=False)
        image_features = self.vision_features_extractor.forward(image_patches, bbox_coords)
        
        if not self.text_pos_emb_2d:
            # process text if 1D positional embeddings are being used
            caption = self.tokenizer(caption, padding=True, return_tensors="pt").to(DEVICE)
        text_features = self.text_features_extractor.forward(caption)
            
        return image_features, text_features

    def forward(self, images, texts, image_folder=None, multi_threading=False, raw_embeds=False):
        image_embeddings = []
        text_embeddings = []
        
        if not raw_embeds: # generate embeddings
            if multi_threading:
                max_workers = min(len(images), 4) # max workers for ThreadPoolExecutor
                print(f'Running in parallel with {max_workers} workers')
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_embeds = {executor.submit(self.generate_image_text_embeddings, image, caption, image_folder): (image, caption) for image, caption in zip(images, texts)}
                    
                    for future in as_completed(future_to_embeds):
                        try:
                            image_embedding, text_embedding = future.result()
                            image_embeddings.append(image_embedding)
                            text_embeddings.append(text_embedding)
                        except Exception as e:
                            print(f"Exception: {e}")
            else:
                for image, caption in zip(images, texts):
                    image_embedding, text_embedding = self.generate_image_text_embeddings(image, caption, image_folder)
                    image_embeddings.append(image_embedding)
                    text_embeddings.append(text_embedding)
        else:
            # raw embeddings in images and text
            image_embeddings = [images]
            text_embeddings = [texts]

        image_embeddings = torch.cat(image_embeddings, dim=0).to(self.device)
        text_embeddings = torch.cat(text_embeddings, dim=0).to(self.device)

        # Apply L2 Normalisation to normalize the embeddings to unit length
        image_embeddings = image_embeddings / image_embeddings.norm(p=2, dim=-1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(p=2, dim=-1, keepdim=True)

        # Compute cosine similarity as logits
        logit_scale = self.model.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeddings, image_embeddings.t()) * logit_scale
        logits_per_image = logits_per_text.t()

        return logits_per_image, logits_per_text

    def upload_model_to_hf_hub(self, model_name, hf_name):
        api = HfApi()

        repo_exists = api.repo_exists(repo_id=f"{hf_name}/{model_name}", token=HF_TOKEN)
        print(f"Repository exists: {repo_exists}")

        if repo_exists:
            repo = Repository(local_dir=model_name, clone_from=f"https://huggingface.co/{hf_name}/{model_name}", token=HF_TOKEN)
            commit_message = "Update model files"
        else:
            repo_url = api.create_repo(repo_id=model_name, token=HF_TOKEN, private=True)
            repo = Repository(local_dir=model_name, clone_from=repo_url, use_auth_token=HF_TOKEN)
            commit_message = "Add model files"

        self.model.save_pretrained(model_name)
        self.tokenizer.save_pretrained(model_name)
        self.processor.save_pretrained(model_name)
        repo.push_to_hub(commit_message=commit_message)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_name", type=str, help="image file name")
    parser.add_argument("--data_name", type=str, help="data directory name")
    parser.add_argument("--pool_type", type=str, default="mean", help="pooling type; options: ['mean', 'cls', 'attention'], default: 'attention'")
    parser.add_argument("--projection_dim", type=int, default=None, help="custom projection dimension, default: 512")
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32", help="CLIP model name, default: openai/clip-vit-base-patch32")
    parser.add_argument("--text", type=str, default="a photo of a dog", help="text input")
    parser.add_argument("--text_pos_emb_2d", action="store_false", help="Use 2D positional embeddings for text")
    args = parser.parse_args()

    semclip = SemCLIP(
        model_name=args.model_name, 
        pool_type=args.pool_type, 
        projection_dim=args.projection_dim, 
        device=DEVICE, 
        text_pos_emb_2d=args.text_pos_emb_2d,
    )
    image_features = semclip.get_image_features(args.image_name, args.data_name)
    text_features = semclip.get_text_features(args.text)
    
    logits_per_image, logits_per_text = semclip(images=image_features, texts=text_features, raw_embeds=True)
    
    print(f'semclip logits_per_image: {logits_per_image}')
    print(f'semclip logits_per_text: {logits_per_text}')
    