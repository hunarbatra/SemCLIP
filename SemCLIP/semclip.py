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


class SemCLIP(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", pool_type='attention', projection_dim=None, device=DEVICE):
        super(SemCLIP, self).__init__() # initialize nn.Module
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name, token=HF_TOKEN).to(device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.patch_processor = CLIPImageProcessor(do_resize=False, do_center_crop=False)  # disable image resizing and center cropping
        self.vision_config = CLIPVisionConfig.from_pretrained(model_name)
        self.text_config = CLIPTextConfig.from_pretrained(model_name)
        self.pool_type = pool_type
        
        if projection_dim is not None:
            self.vision_config.projection_dim = projection_dim
            self.text_config.projection_dim = projection_dim

        self.vision_model = SemCLIPVision(self.model, self.vision_config, self.pool_type).to(device)
        self.text_model = SemCLIPText(self.model, self.text_config, self.tokenizer, self.pool_type).to(device)
        
        self.text_projection = nn.Linear(self.text_config.hidden_size, self.text_config.projection_dim, bias=False).to(DEVICE)
        self.visual_projection = nn.Linear(self.vision_config.hidden_size, self.vision_config.projection_dim, bias=False).to(DEVICE)

    def get_image_features(self, image_name: str, data_name: str, image_file: Optional[np.ndarray] = None):
        # load image segments patches and normalized bounding box coordinates
        image_resized_patches, normalized_bbox_coords = preprocess_patches(image_name, data_name, image_file=image_file)
        # preprocess the patches (CLIPImageProcessor)
        image_resized_patches = convert_patches_to_pixel_values(image_resized_patches, patch_size=self.vision_config.patch_size, patch_processor=self.patch_processor)
        
        pooled_output = self.vision_model(image_resized_patches, normalized_bbox_coords)
        
        # pass the pooled output through a final linear projection layer
        image_embeds = self.visual_projection(pooled_output)
        
        # image features
        return image_embeds

    def get_text_features(self, text):
        # applies text processor, generates SemCLIPTextEmbeddings with 2D positional embeddings, and passes through the CLIP transformer encoder followed by pooling 
        pooled_output = self.text_model(text)
        
        # Pass the pooled output through a final linear projection
        text_embeds = self.text_projection(pooled_output)

        # text features
        return text_embeds
    
    def process_image_text(self, image, caption, image_folder=None):
        if image_folder is None:
            image_embedding = self.get_image_features(image_name=None, data_name=None, image_file=image)
        else:
            image_embedding = self.get_image_features(image_name=image, data_name=image_folder)
        text_embedding = self.get_text_features(text=caption)
        return image_embedding, text_embedding

    def get_semclip_embeddings(self, images, captions, images_folder, multi_threading=False):
        image_embeddings = []
        text_embeddings = []
        
        if multi_threading:
            max_workers = min(len(images), 64) # max workers for ThreadPoolExecutor
            print(f'Running in parallel with {max_workers} workers')

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_embeds = {executor.submit(self.process_image_text, image, caption, images_folder): (image, caption) for image, caption in zip(images, captions)}
                
                for future in as_completed(future_to_embeds):
                    try:
                        image_embedding, text_embedding = future.result()
                        image_embeddings.append(image_embedding)
                        text_embeddings.append(text_embedding)
                    except Exception as e:
                        print(f"Exception: {e}")
        else:
            for image, caption in zip(images, captions):
                image_embedding = self.get_image_features(image_name=image, data_name=images_folder)
                text_embedding = self.get_text_features(text=caption)

                image_embeddings.append(image_embedding)
                text_embeddings.append(text_embedding)
                
        image_embeddings = torch.cat(image_embeddings, dim=0)
        text_embeddings = torch.cat(text_embeddings, dim=0)
        
        image_embeddings.to(self.device)
        text_embeddings.to(self.device)

        return image_embeddings, text_embeddings

    def get_semclip_embeddings_direct_img(self, images, captions, multi_threading=False):
        image_embeddings = []
        text_embeddings = []
        
        if multi_threading:
            max_workers = min(len(images), 64) # max workers for ThreadPoolExecutor
            print(f'Running in parallel with {max_workers} workers')
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_embeds = {executor.submit(self.process_image_text, image, caption): (image, caption) for image, caption in zip(images, captions)}
                
                for future in as_completed(future_to_embeds):
                    try:
                        image_embedding, text_embedding = future.result()
                        image_embeddings.append(image_embedding)
                        text_embeddings.append(text_embedding)
                    except Exception as e:
                        print(f"Exception: {e}")
        else:
            for image, caption in zip(images, captions):
                image_embedding = self.get_image_features(image_name=None, data_name=None, image_file=image)
                text_embedding = self.get_text_features(text=caption)

                image_embeddings.append(image_embedding)
                text_embeddings.append(text_embedding)

        image_embeddings = torch.cat(image_embeddings, dim=0)
        text_embeddings = torch.cat(text_embeddings, dim=0)
        
        image_embeddings.to(self.device)
        text_embeddings.to(self.device)

        return image_embeddings, text_embeddings

    def forward(self, images, texts, image_folder=None, raw_embeds=False, multi_threading=False):
        if not raw_embeds:
            if image_folder is None:
                image_embeddings, text_embeddings = self.get_semclip_embeddings_direct_img(images, texts, multi_threading)
            else:
                image_embeddings, text_embeddings = self.get_semclip_embeddings(images, texts, image_folder, multi_threading)
        else:
            image_embeddings = images
            text_embeddings = texts
            
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
    args = parser.parse_args()

    semclip = SemCLIP(model_name=args.model_name, pool_type=args.pool_type, projection_dim=args.projection_dim, device=DEVICE)
    image_features = semclip.get_image_features(args.image_name, args.data_name)
    text_features = semclip.get_text_features(args.text)
    
    logits_per_image, logits_per_text = semclip(images=image_features, texts=text_features, raw_embeds=True)
    
    print(f'semclip logits_per_image: {logits_per_image}')
    print(f'semclip logits_per_text: {logits_per_text}')
    