import os
import torch
import argparse

import numpy as np
import pandas as pd

from SemCLIP.semclip import SemCLIP
from SemCLIP.image_utils import DEVICE, create_batches, pil_to_cv2
from config import model_mapper, dataset_mapper, dataset_keys

from transformers import CLIPProcessor, CLIPModel, AutoModel, AutoProcessor, AutoTokenizer, AutoImageProcessor, Dinov2ForImageClassification
from datasets import load_dataset
from typing import Optional
from PIL import Image
from tqdm import tqdm


def compute_zeroshot_and_cosine_scores(preds, logits_per_image, captions_batch, images_names, results, data, model_name):
    for i, pred_idx in enumerate(preds):
        is_correct = int(captions_batch[pred_idx] == captions_batch[i]) # check if predicted caption == true caption
        cosine_score = float(logits_per_image[i][pred_idx]) # get the cosine score for each image for the predicted class (caption)
        results.append({
            "image_name": images_names[i], 
            "data_name": dataset_mapper[data],
            "model_name": model_mapper[model_name], 
            "caption": captions_batch[i], 
            "prediction": captions_batch,
            "caption_idx": i,
            "prediction_idx": pred_idx,
            "is_correct": is_correct,
            "cosine_score": cosine_score,
        })
            
    return results

def compute_scores(results_df):
    accuracy = np.mean(results_df['is_correct'])
    accuracy_std_err = np.std(results_df["is_correct"]) / np.sqrt(len(results_df["is_correct"]))
    cosine_score = np.mean(results_df["cosine_score"])
    cosine_score_std_err = np.std(results_df["cosine_score"]) / np.sqrt(len(results_df["cosine_score"]))
    
    return (accuracy, accuracy_std_err, cosine_score, cosine_score_std_err)

def print_scores(results_df, accuracy, accuracy_std_err, cosine_score, cosine_score_std_err, batch_idx=None):
    prefix = f'Batch {batch_idx}:' if batch_idx is not None else ''
    print(f'{prefix} Zero-shot top-1 Accuracy: {accuracy * 100:.2f}%, std err: {accuracy_std_err * 100:.2f}%')
    print(f'{prefix} Avg Cosine Similarity Score: {np.mean(results_df["cosine_score"])}, std err: {cosine_score_std_err:.2f}%')

def save_scores(results, eval_run_name, data, model_name):
    results_df = pd.DataFrame(results)
    
    accuracy, accuracy_std_err, cosine_score, cosine_score_std_err = compute_scores(results_df)
    print_scores(results_df, accuracy, accuracy_std_err, cosine_score, cosine_score_std_err)
    
    os.makedirs(f'experiments/{eval_run_name}', exist_ok=True)
    file_name = model_mapper[model_name].split('/')[-1]
    results_df.to_csv(f'experiments/{eval_run_name}/{data}_{file_name}.csv')
    
def load_scores(eval_run_name, data, model_name):
    results_df = pd.read_csv(f'experiments/{eval_run_name}/{data}_{model_name}.csv')
    
    accuracy, accuracy_std_err, cosine_score, cosine_score_std_err = compute_scores(results_df)
    print_scores(results_df, accuracy, accuracy_std_err, cosine_score, cosine_score_std_err)
    
    return accuracy, accuracy_std_err, cosine_score, cosine_score_std_err

def evaluate_semclip_model(
    # SemCLIP related parameters
    model_name: str = 'semclip-v4', # base model i.e SemCLIP wrapper over CLIP, set to semclip variant from config to evaluate finetuned SemCLIP
    pool_type: str = 'attention',
    projection_dim: int = 512,
    multi_threading: bool = False,
    text_pos_emb_2d: bool = False,
    # evaluation parameters
    data: str = 'COCO-13k',
    batch_size: int = 64,
    data_split: str = 'validation',
    data_subset: Optional[int] = None,
    eval_run_name: str = 'test',
    use_finetuned: bool = False,
    max_batch: bool = False, #  to get names of all the classes in the dataset as the set of potential text pairings and predict the most probable -- CLIP Zero-shot eval
    verbose: bool = False, # print semclip weights and shape to double check,
):
    # Load SemCLIP model
    fine_tuned_model = model_name if use_finetuned else None
    semclip = SemCLIP(
        model_name=model_mapper[model_name],
        pool_type=pool_type,
        projection_dim=projection_dim,
        device=DEVICE,
        text_pos_emb_2d=text_pos_emb_2d, # 1D positional embeddings for text by default
        ignore_mismatched_sizes=True if model_name.startswith('semclip') else False,
        fine_tuned_model=fine_tuned_model,
        verbose=verbose,
    ).to(DEVICE)
    
    # Load the dataset
    dataset = load_dataset(dataset_mapper[data])
    if data_subset:
        dataset = dataset[data_split].select(range(data_subset))
    else:
        dataset = dataset[data_split]
    test_loader = create_batches(dataset, batch_size)
        
    results = []
    batch_size = batch_size if not max_batch else len(dataset)
    pbar = tqdm(test_loader, total=len(dataset) // batch_size)
    
    data_img_key = dataset_keys[data]['img']
    data_text_key = dataset_keys[data]['text']
    data_img_idx_key = dataset_keys[data]['img_idx']
    
    semclip.model.eval() # Set the model to evaluation mode
    
    for batch_idx, batch in enumerate(pbar):
        predictions = []
        
        # Get the images and captions for the batch
        image_batch_pil = batch[data_img_key]
        captions_batch = batch[data_text_key]
        images_names = batch[data_img_idx_key]
        
        # Convert the batch of PIL images to OpenCV images
        image_batch_cv2 = [pil_to_cv2(img) for img in image_batch_pil]
        print(f'image_batch_cv2[0].shape: {image_batch_cv2[0].shape}')
        
        with torch.no_grad():
            # Get the image features
            logits_per_image, logits_per_text = semclip(
                images=image_batch_cv2, 
                texts=captions_batch, 
                multi_threading=multi_threading
            )
        
        probs = logits_per_image.softmax(dim=1)
        preds = probs.argmax(dim=1) # get the predicted class index for each image
        
        results = compute_zeroshot_and_cosine_scores(preds, logits_per_image, captions_batch, images_names, results, data, model_name)
        curr_results_df = pd.DataFrame(results)
        accuracy, accuracy_std_err, cosine_score, cosine_score_std_err = compute_scores(curr_results_df)
        print_scores(curr_results_df, accuracy, accuracy_std_err, cosine_score, cosine_score_std_err, batch_idx)
        
    save_scores(results, eval_run_name, data, model_name)
    
def evaluate_clip_model(
    model_name: str = 'clip',
    batch_size: int = 64,
    data: str = 'COCO-13k',
    data_split: str = 'validation',
    data_subset: Optional[int] = None,
    eval_run_name: str = 'text',
    max_batch: bool = False, #  to get names of all the classes in the dataset as the set of potential text pairings and predict the most probable -- CLIP Zero-shot eval
):
    # Load the model and processor
    model = AutoModel.from_pretrained(model_mapper[model_name]).to(DEVICE)
    processor = AutoProcessor.from_pretrained(model_mapper[model_name])
    
    # Load the dataset
    dataset = load_dataset(dataset_mapper[data])
    if data_subset:
        dataset = dataset[data_split].select(range(data_subset))
    else:
        dataset = dataset[data_split]
    test_loader = create_batches(dataset, batch_size)
    
    results = []
    batch_size = batch_size if not max_batch else len(dataset)
    pbar = tqdm(test_loader, total=len(dataset) // batch_size)
    
    data_img_key = dataset_keys[data]['img']
    data_text_key = dataset_keys[data]['text']
    data_img_idx_key = dataset_keys[data]['img_idx']
    
    model.eval() # Set the model to evaluation mode
    
    for batch_idx, batch in enumerate(pbar):
        predictions = []
        
        # Get the images and captions for the batch
        images_batch = batch[data_img_key]
        captions_batch = batch[data_text_key]
        images_names = batch[data_img_idx_key]
        
        images_batch_cv2 = [pil_to_cv2(img) for img in images_batch]
        
        inputs = processor(text=captions_batch, images=images_batch_cv2, return_tensors="pt", padding=True).to(DEVICE)
        outputs = model(**inputs)
        
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        preds = probs.argmax(dim=1) # get the predicted class index for each image
        
        print(f'logits_per_image: {logits_per_image}')
        print(f'logits_per_image.shape: {logits_per_image.shape}')
                
        results = compute_zeroshot_and_cosine_scores(preds, logits_per_image, captions_batch, images_names, results, data, model_name)
        curr_results_df = pd.DataFrame(results)
        accuracy, accuracy_std_err, cosine_score, cosine_score_std_err = compute_scores(curr_results_df)
        print_scores(curr_results_df, accuracy, accuracy_std_err, cosine_score, cosine_score_std_err, batch_idx)
                
    save_scores(results, eval_run_name, data, model_name)

def evaluate_siglip_model(
    model_name: str = 'siglip',
    batch_size: int = 64,
    data: str = 'COCO-13k',
    data_split: str = 'validation',
    data_subset: Optional[int] = None,
    eval_run_name: str = 'text',
    max_batch: bool = False, #  to get names of all the classes in the dataset as the set of potential text pairings and predict the most probable -- CLIP Zero-shot eval
):
    # Load the model and processor
    model = AutoModel.from_pretrained(model_mapper[model_name]).to(DEVICE)
    processor = AutoProcessor.from_pretrained(model_mapper[model_name])
    
    # Load the dataset
    dataset = load_dataset(dataset_mapper[data])
    if data_subset:
        dataset = dataset[data_split].select(range(data_subset))
    else:
        dataset = dataset[data_split]
    test_loader = create_batches(dataset, batch_size)
    
    results = []
    batch_size = batch_size if not max_batch else len(dataset)
    pbar = tqdm(test_loader, total=len(dataset) // batch_size)
    
    data_img_key = dataset_keys[data]['img']
    data_text_key = dataset_keys[data]['text']
    data_img_idx_key = dataset_keys[data]['img_idx']
    
    model.eval() # Set the model to evaluation mode
    
    for batch_idx, batch in enumerate(pbar):
        predictions = []
        
        # Get the images and captions for the batch
        images_batch = batch[data_img_key]
        captions_batch = batch[data_text_key]
        images_names = batch[data_img_idx_key]
        
        images_batch_cv2 = [pil_to_cv2(img) for img in images_batch]
        
        inputs = processor(text=captions_batch, images=images_batch_cv2, return_tensors="pt", padding="max_length").to(DEVICE)
        outputs = model(**inputs)
        
        logits_per_image = outputs.logits_per_image
        probs = torch.sigmoid(logits_per_image)
        preds = probs.argmax(dim=1) # get the predicted class index for each image
        
        results = compute_zeroshot_and_cosine_scores(preds, logits_per_image, captions_batch, images_names, results, data, model_name)
        curr_results_df = pd.DataFrame(results)
        accuracy, accuracy_std_err, cosine_score, cosine_score_std_err = compute_scores(curr_results_df)
        print_scores(curr_results_df, accuracy, accuracy_std_err, cosine_score, cosine_score_std_err, batch_idx)
        
    save_scores(results, eval_run_name, data, model_name)

def evaluate_dinov2_model(
    model_name: str = 'dinov2',
    batch_size: int = 64,
    data: str = 'COCO-13k',
    data_split: str = 'validation',
    data_subset: Optional[int] = None,
    eval_run_name: str = 'text',
    max_batch: bool = False, #  to get names of all the classes in the dataset as the set of potential text pairings and predict the most probable -- CLIP Zero-shot eval
):
    assert data == 'imagenet1k-val', "Dinov2 model only supports imagenet1k-val dataset"
    # Load the model and processor
    model = Dinov2ForImageClassification.from_pretrained(model_mapper[model_name]).to(DEVICE)
    processor = AutoImageProcessor.from_pretrained(model_mapper[model_name])
    
    # Load the dataset
    dataset = load_dataset(dataset_mapper[data])
    if data_subset:
        dataset = dataset[data_split].select(range(data_subset))
    else:
        dataset = dataset[data_split]
    test_loader = create_batches(dataset, batch_size)
    
    results = []
    batch_size = batch_size if not max_batch else len(dataset)
    pbar = tqdm(test_loader, total=len(dataset) // batch_size)
    
    data_img_key = dataset_keys[data]['img']
    data_text_key = dataset_keys[data]['text']
    data_img_idx_key = dataset_keys[data]['img_idx']
    
    model.eval() # Set the model to evaluation mode
    
    for batch_idx, batch in enumerate(pbar):
        predictions = []
        
        # Get the images and captions for the batch
        images_batch = batch[data_img_key]
        images_names = batch[data_img_idx_key]
        
        images_batch_cv2 = [pil_to_cv2(img) for img in images_batch]
        
        inputs = processor(images=images_batch_cv2, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            logits_per_image = model(**inputs).logits
        
        print(f'logits_per_image: {logits_per_image}')
        print(f'logits_per_image.shape: {logits_per_image.shape}')
        # probs = logits_per_image.softmax(dim=1)
        preds = logits_per_image.argmax(dim=-1) # get the predicted class index for each image
             
        for i, pred_idx in enumerate(preds):
            pred_idx = pred_idx.item()
            is_correct = int(model.config.id2label[pred_idx] == model.config.id2label[i]) # check if predicted caption == true caption
            cosine_score = float(logits_per_image[i][pred_idx]) # get the cosine score for each image for the predicted class (caption)
            results.append({
                "image_name": images_names[i],
                "data_name": dataset_mapper[data],
                "model_name": model_mapper[model_name],
                "caption": model.config.id2label[i],
                "prediction": model.config.id2label,
                "caption_idx": i,
                "prediction_idx": pred_idx,
                "is_correct": is_correct,
                "cosine_score": cosine_score,
            })
        curr_results_df = pd.DataFrame(results)
        accuracy, accuracy_std_err, cosine_score, cosine_score_std_err = compute_scores(curr_results_df)
        print_scores(curr_results_df, accuracy, accuracy_std_err, cosine_score, cosine_score_std_err, batch_idx)
                
    save_scores(results, eval_run_name, data, model_name)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a model on a dataset')
    parser.add_argument('--model_name', type=str, default='clip', help='Model to evaluate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for evaluation')
    parser.add_argument('--data', type=str, default='COCO-13k', help='Dataset to evaluate on')
    parser.add_argument('--data_split', type=str, default='validation', help='Dataset split to evaluate on')
    parser.add_argument('--data_subset', type=int, default=None, help='Number of subset samples to evaluate on')
    parser.add_argument('--eval_run_name', type=str, default='text', help='Name of the evaluation run')
    parser.add_argument('--pool_type', type=str, default='attention', help='Pooling type for SemCLIP')
    parser.add_argument('--projection_dim', type=int, default=512, help='Projection dimension for SemCLIP')
    parser.add_argument('--multi_threading', action='store_true', help='Use multi-threading for SemCLIP')
    parser.add_argument('--text_pos_emb_2d', action='store_true', help='Use 2D positional embeddings for text in SemCLIP')
    parser.add_argument('--use_finetuned', action='store_true', help='Use fine-tuned SemCLIP model')
    parser.add_argument('--verbose', action='store_true', help='Print model weights and shapes') # print semclip updated layers weights and shape to double check
    parser.add_argument('--cosine', action='store_true', help='Use direct cosine similarity for evaluation')
    parser.add_argument('--max_batch', action='store_true', help='Use the dataset length as the batch size')
    
    args = parser.parse_args()
    
    if args.model_name == 'clip':
        evaluate_clip_model(
            model_name=args.model_name,
            batch_size=args.batch_size,
            data=args.data,
            data_split=args.data_split,
            data_subset=args.data_subset,
            eval_run_name=args.eval_run_name,
            max_batch=args.max_batch,
        )
    elif args.model_name == 'dinov2':
        evaluate_dinov2_model(
            model_name=args.model_name,
            batch_size=args.batch_size,
            data=args.data,
            data_split=args.data_split,
            data_subset=args.data_subset,
            eval_run_name=args.eval_run_name,
            max_batch=args.max_batch,
        )
    elif args.model_name == 'siglip':
        evaluate_siglip_model(
            model_name=args.model_name,
            batch_size=args.batch_size,
            data=args.data,
            data_split=args.data_split,
            data_subset=args.data_subset,
            eval_run_name=args.eval_run_name,
            max_batch=args.max_batch,
        )   
    elif args.model_name.startswith('semclip'):
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
            use_finetuned=args.use_finetuned,
            verbose=args.verbose,
            max_batch=args.max_batch,
        )
    else:
        raise ValueError(f'Invalid model name: {args.model_name}, set HuggingFace model config in config.py')
    