import os
import torch
import argparse

import numpy as np
import pandas as pd

from SemCLIP.semclip import SemCLIP
from SemCLIP.image_utils import DEVICE, create_batches, pil_to_cv2
from config import model_mapper, dataset_mapper

from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset
from typing import Optional
from PIL import Image
from tqdm import tqdm


def evaluate_semclip_model(
    # SemCLIP related parameters
    model_name: str = 'clip', # base model i.e SemCLIP wrapper over CLIP, set to semclip variant from config to evaluate finetuned SemCLIP
    pool_type: str = 'attention',
    projection_dim: int = 512,
    multi_threading: bool = False,
    text_pos_emb_2d: bool = False,
    # evaluation parameters
    data: str = 'COCO-13k',
    batch_size: int = 64,
    data_split: str = 'test',
    data_subset: Optional[int] = None,
    eval_run_name: str = 'text',
):
    # Load SemCLIP model
    semclip = SemCLIP(
        model_name=model_mapper[model_name],
        pool_type=pool_type,
        projection_dim=projection_dim,
        device=DEVICE,
        text_pos_emb_2d=text_pos_emb_2d, # 1D positional embeddings for text by default
        ignore_mismatched_sizes=True if model_name.startswith('semclip') else False,
    ).to(DEVICE)
    
    # Load the dataset
    dataset = load_dataset(dataset_mapper[data])
    if data_subset:
        dataset = dataset[data_split].select(range(data_subset))
    else:
        dataset = dataset[data_split]
    test_loader = create_batches(dataset, batch_size)
        
    results = []
    pbar = tqdm(test_loader, total=len(dataset) // batch_size)
    
    semclip.model.eval() # Set the model to evaluation mode
    
    for batch_idx, batch in enumerate(pbar):
        predictions = []
        
        # Get the images and captions for the batch
        image_batch_pil = batch['downloaded_img']
        captions_batch = batch['caption']
        images_names = batch['image']
        
        # Convert the batch of PIL images to OpenCV images
        image_batch_cv2 = [pil_to_cv2(img) for img in image_batch_pil]
        
        with torch.no_grad():
            # Get the image features
            image_features, label_features = semclip(
                images=image_batch_cv2, 
                texts=captions_batch, 
                multi_threading=multi_threading,
                return_features=True
            )
            
            # Convert the features list[torch.tensor] to numpy arrays
            # remove the extra dim [1, 1, 512] -> [1, 512] with squeeze()
            image_features = np.array([img_feat.detach().to(DEVICE).squeeze().numpy() for img_feat in image_features])
            label_features = np.array([label_feat.detach().to(DEVICE).squeeze().numpy() for label_feat in label_features])
            
            img_emb = image_features / np.linalg.norm(image_features, axis=1, keepdims=True)
            label_emb = label_features / np.linalg.norm(label_features, axis=1, keepdims=True)
        
        # Compute the similarity scores
        scores = np.dot(img_emb, label_emb.T)
        predictions.append(np.argmax(scores, axis=1))
        
        # Compute the number of correct predictions
        for i, pred_idx in enumerate(predictions[0]):
            is_correct = int(captions_batch[pred_idx] == captions_batch[i])
            results.append({
                "image_name": images_names[i], 
                "data_name": dataset_mapper[data], 
                "model_name": model_mapper[model_name],
                "caption": captions_batch[i], 
                "prediction": captions_batch, 
                "caption_idx": i, 
                "prediction_idx": pred_idx, 
                "is_correct": is_correct
            })
                
        print(f'Correct predictions % so far: {np.mean([r["is_correct"] for r in results])}')
        
    results_df = pd.DataFrame(results)
    
    accuracy = np.mean(results_df['is_correct'])
    print(f'Zero-shot top-1 Accuracy: {accuracy}')
    
    os.makedirs(f'experiments/{eval_run_name}', exist_ok=True)
    file_name = model_mapper[model_name].split('/')[-1]
    results_df.to_csv(f'experiments/{eval_run_name}/{file_name}.csv')

def evaluate_clip_model(
    model_name: str = 'clip',
    batch_size: int = 64,
    data: str = 'COCO-13k',
    data_split: str = 'test',
    data_subset: Optional[int] = None,
    eval_run_name: str = 'text',
):
    # Load the model and processor
    model = CLIPModel.from_pretrained(model_mapper[model_name]).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(model_mapper[model_name])
    
    # Load the dataset
    dataset = load_dataset(dataset_mapper[data])
    if data_subset:
        dataset = dataset[data_split].select(range(data_subset))
    else:
        dataset = dataset[data_split]
    test_loader = create_batches(dataset, batch_size)
    
    results = []
    pbar = tqdm(test_loader, total=len(dataset) // batch_size)
    
    model.eval() # Set the model to evaluation mode
    
    for batch_idx, batch in enumerate(pbar):
        predictions = []
        
        # Get the images and captions for the batch
        images_batch = batch['downloaded_img']
        captions_batch = batch['caption']
        images_names = batch['image']
        
        with torch.no_grad():
            # Get the image features
            image_inputs = processor(text=None, images=images_batch, return_tensors="pt")['pixel_values'].to(DEVICE)
            img_emb = model.get_image_features(image_inputs).detach().to(DEVICE).numpy()
            img_emb = img_emb / np.linalg.norm(img_emb, axis=1, keepdims=True)
            
            # Get the text features
            label_inputs = processor(text=captions_batch, images=None, return_tensors="pt", padding=True).to(DEVICE)
            label_emb = model.get_text_features(**label_inputs).detach().to(DEVICE).numpy()
            label_emb = label_emb / np.linalg.norm(label_emb, axis=1, keepdims=True)
            
        # Compute the similarity scores
        scores = np.dot(img_emb, label_emb.T)
        predictions.append(np.argmax(scores, axis=1))
        
        # Compute the number of correct predictions
        for i, pred_idx in enumerate(predictions[0]):
            is_correct = int(captions_batch[pred_idx] == captions_batch[i])
            results.append({
                "image_name": images_names[i], 
                "data_name": dataset_mapper[data], 
                "model_name": model_mapper[model_name],
                "caption": captions_batch[i], 
                "prediction": captions_batch, 
                "caption_idx": i, 
                "prediction_idx": pred_idx, 
                "is_correct": is_correct
            })
                
        print(f'Correct predictions % so far: {np.mean([r["is_correct"] for r in results])}')
        
    results_df = pd.DataFrame(results)
    
    accuracy = np.mean(results_df['is_correct'])
    print(f'Zero-shot top-1 Accuracy: {accuracy}')
    
    os.makedirs(f'experiments/{eval_run_name}', exist_ok=True)
    file_name = model_mapper[model_name].split('/')[-1]
    results_df.to_csv(f'experiments/{eval_run_name}/{file_name}.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a model on a dataset')
    parser.add_argument('--semclip', action='store_true', help='Evaluate SemCLIP model')
    parser.add_argument('--model_name', type=str, default='clip', help='Model to evaluate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for evaluation')
    parser.add_argument('--data', type=str, default='COCO-13k', help='Dataset to evaluate on')
    parser.add_argument('--data_split', type=str, default='test', help='Dataset split to evaluate on')
    parser.add_argument('--data_subset', type=int, default=None, help='Number of subset samples to evaluate on')
    parser.add_argument('--eval_run_name', type=str, default='text', help='Name of the evaluation run')
    parser.add_argument('--pool_type', type=str, default='attention', help='Pooling type for SemCLIP')
    parser.add_argument('--projection_dim', type=int, default=512, help='Projection dimension for SemCLIP')
    parser.add_argument('--multi_threading', action='store_true', help='Use multi-threading for SemCLIP')
    parser.add_argument('--text_pos_emb_2d', action='store_true', help='Use 2D positional embeddings for text in SemCLIP')
    
    args = parser.parse_args()
    
    if not args.semclip:
        evaluate_clip_model(
            model_name=args.model_name,
            batch_size=args.batch_size,
            data=args.data,
            data_split=args.data_split,
            data_subset=args.data_subset,
            eval_run_name=args.eval_run_name,
        )
    else:
        evaluate_semclip_model(
            model_name=args.model_name,
            pool_type=args.pool_type,
            projection_dim=args.projection_dim,
            multi_threading=args.multi_threading,
            text_pos_emb_2d=args.text_pos_emb_2d,
            data=args.data,
            batch_size=args.batch_size,
            data_split=args.data_split,
            data_subset=args.data_subset,
            eval_run_name=args.eval_run_name,
        )