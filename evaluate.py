import os
import torch
import argparse

import numpy as np
import pandas as pd

from SemCLIP.semclip import SemCLIP
from SemCLIP.image_utils import DEVICE, create_batches, pil_to_cv2
from config import model_mapper, dataset_mapper, dataset_keys

from transformers import CLIPProcessor, CLIPModel, AutoModel, AutoProcessor, AutoTokenizer
from datasets import load_dataset
from typing import Optional
from PIL import Image
from tqdm import tqdm


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
    data_split: str = 'test',
    data_subset: Optional[int] = None,
    eval_run_name: str = 'text',
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
        images_names = batch[image_idx_key]
        
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
            image_features = np.array([img_feat.detach().cpu().squeeze().numpy() for img_feat in image_features])
            label_features = np.array([label_feat.detach().cpu().squeeze().numpy() for label_feat in label_features])
            
            img_emb = image_features / np.linalg.norm(image_features, axis=1, keepdims=True)
            label_emb = label_features / np.linalg.norm(label_features, axis=1, keepdims=True)
        
        # Compute the similarity scores
        scores = np.dot(img_emb, label_emb.T)
        predictions.append(np.argmax(scores, axis=1))
        
        # Compute the number of correct predictions
        for i, pred_idx in enumerate(predictions[0]):
            is_correct = int(captions_batch[pred_idx] == captions_batch[i])
            cosine_score = float(scores[i][pred_idx])
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
                
        print(f'Correct predictions % so far: {np.mean([r["is_correct"] for r in results])}')
        print(f'Avg Cosine Similarity Score so far: {np.mean([r["cosine_score"] for r in results])}')
        
    results_df = pd.DataFrame(results)
    
    accuracy = np.mean(results_df['is_correct'])
    print(f'Zero-shot top-1 Accuracy: {accuracy}, std: {np.std(results_df["is_correct"])}')
    print(f'Avg Cosine Similarity Score: {np.mean(results_df["cosine_score"])}, std: {np.std(results_df["cosine_score"])}')
    
    os.makedirs(f'experiments/{eval_run_name}', exist_ok=True)
    file_name = model_mapper[model_name].split('/')[-1]
    results_df.to_csv(f'experiments/{eval_run_name}/{data}_{file_name}.csv')
    
# def evaluate_semclip_model_cosine(
#     # SemCLIP related parameters
#     model_name: str = 'semclip-v4', # base model i.e SemCLIP wrapper over CLIP, set to semclip variant from config to evaluate finetuned SemCLIP
#     pool_type: str = 'attention',
#     projection_dim: int = 512,
#     multi_threading: bool = False,
#     text_pos_emb_2d: bool = False,
#     # evaluation parameters
#     data: str = 'COCO-13k',
#     batch_size: int = 64,
#     data_split: str = 'test',
#     data_subset: Optional[int] = None,
#     eval_run_name: str = 'text',
#     use_finetuned: bool = False,
#     verbose: bool = False, # print semclip weights and shape to double check
# ):
#     # Load SemCLIP model
#     fine_tuned_model = model_name if use_finetuned else None
#     semclip = SemCLIP(
#         model_name=model_mapper[model_name],
#         pool_type=pool_type,
#         projection_dim=projection_dim,
#         device=DEVICE,
#         text_pos_emb_2d=text_pos_emb_2d, # 1D positional embeddings for text by default
#         ignore_mismatched_sizes=True if model_name.startswith('semclip') else False,
#         fine_tuned_model=fine_tuned_model,
#         verbose=verbose,
#     ).to(DEVICE)
    
#     # Load the dataset
#     dataset = load_dataset(dataset_mapper[data])
#     if data_subset:
#         dataset = dataset[data_split].select(range(data_subset))
#     else:
#         dataset = dataset[data_split]
#     test_loader = create_batches(dataset, batch_size)
        
#     results = []
#     pbar = tqdm(test_loader, total=len(dataset) // batch_size)
    
#     data_img_key = dataset_keys[data]['img']
#     data_text_key = dataset_keys[data]['text']
#     data_img_idx_key = dataset_keys[data]['img_idx']
    
#     semclip.model.eval() # Set the model to evaluation mode
    
#     for batch_idx, batch in enumerate(pbar):
#         predictions = []
        
#         # Get the images and captions for the batch
#         image_batch_pil = batch[data_img_key]
#         captions_batch = batch[data_text_key]
#         images_names = batch[data_img_idx_key]
        
#         # Convert the batch of PIL images to OpenCV images
#         image_batch_cv2 = [pil_to_cv2(img) for img in image_batch_pil]
        
#         with torch.no_grad():
#             # Get the image features / embeddings
#             image_features, label_features = semclip(
#                 images=image_batch_cv2, 
#                 texts=captions_batch, 
#                 multi_threading=multi_threading,
#                 return_features=True
#             )
            
#             # cosine similarity score between image and text embeddings
#             logits_per_image, logits_per_text = semclip(
#                 images=image_features,
#                 texts=label_features,
#                 raw_embeds=True
#             )
        
#         for i, cos_score in enumerate(logits_per_image):
#             curr_score = float(cos_score[0])
#             results.append({
#                 "image_name": images_names[i], 
#                 "data_name": dataset_mapper[data], 
#                 "model_name": model_mapper[model_name],
#                 "caption": captions_batch[i], 
#                 "cosine_score": curr_score
#             })
                
#         print(f'Avg Cosine Similarity Score so far: {np.mean([r["cosine_score"] for r in results])}')
        
#     results_df = pd.DataFrame(results)
    
#     accuracy = np.mean(results_df['cosine_score'])
#     print(f'Avg Cosine Similarity Score: {accuracy}')
    
#     os.makedirs(f'experiments/{eval_run_name}', exist_ok=True)
#     file_name = model_mapper[model_name].split('/')[-1]
#     results_df.to_csv(f'experiments/{eval_run_name}/{data}_cosine_{file_name}.csv')

def evaluate_clip_model(
    model_name: str = 'clip',
    batch_size: int = 64,
    data: str = 'COCO-13k',
    data_split: str = 'test',
    data_subset: Optional[int] = None,
    eval_run_name: str = 'text',
    max_batch: bool = False, #  to get names of all the classes in the dataset as the set of potential text pairings and predict the most probable -- CLIP Zero-shot eval
):
    # Load the model and processor
    model = AutoModel.from_pretrained(model_mapper[model_name]).to(DEVICE)
    processor = AutoProcessor.from_pretrained(model_mapper[model_name])
    tokenizer = AutoTokenizer.from_pretrained(model_mapper[model_name])
    
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
        
        with torch.no_grad():
            # Get the image features
            image_inputs = processor(images=images_batch_cv2, return_tensors="pt")['pixel_values'].to(DEVICE)
            # img_emb = model.get_image_features(image_inputs).detach().cpu().numpy()
            img_emb = model.get_image_features(image_inputs)
            img_emb = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)
            img_emb = img_emb.detach().cpu().numpy()
            # img_emb = img_emb / np.linalg.norm(img_emb, axis=1, keepdims=True)
            
            # Get the text features
            label_inputs = processor(text=captions_batch, return_tensors="pt", padding="max_length").to(DEVICE)
            # label_emb = model.get_text_features(**label_inputs).detach().cpu().numpy()
            label_emb = model.get_text_features(**label_inputs)
            label_emb = label_emb / label_emb.norm(p=2, dim=-1, keepdim=True)
            label_emb = label_emb.detach().cpu().numpy()
            # label_emb = label_emb / np.linalg.norm(label_emb, axis=1, keepdims=True)
            
        # Compute the similarity scores
        scores = np.dot(img_emb, label_emb.T)
        predictions.append(np.argmax(scores, axis=1))
        
        # Compute the number of correct predictions
        for i, pred_idx in enumerate(predictions[0]):
            is_correct = int(captions_batch[pred_idx] == captions_batch[i])
            cosine_score = float(scores[i][pred_idx])
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
                
        print(f'Correct predictions % so far: {np.mean([r["is_correct"] for r in results])}')
        print(f'Avg Cosine Similarity Score so far: {np.mean([r["cosine_score"] for r in results])}')
        
    results_df = pd.DataFrame(results)
    
    accuracy = np.mean(results_df['is_correct'])
    print(f'Zero-shot top-1 Accuracy: {accuracy}, std: {np.std(results_df["is_correct"])}')
    print(f'Avg Cosine Similarity Score: {np.mean(results_df["cosine_score"])}, std: {np.std(results_df["cosine_score"])}')
    
    os.makedirs(f'experiments/{eval_run_name}', exist_ok=True)
    file_name = model_mapper[model_name].split('/')[-1]
    results_df.to_csv(f'experiments/{eval_run_name}/{data}_{file_name}.csv')
    
# def evaluate_clip_model_cosine(
#     model_name: str = 'clip',
#     batch_size: int = 64,
#     data: str = 'COCO-13k',
#     data_split: str = 'test',
#     data_subset: Optional[int] = None,
#     eval_run_name: str = 'text',
# ):
#     # Load the model and processor
#     model = AutoModel.from_pretrained(model_mapper[model_name]).to(DEVICE)
#     processor = AutoProcessor.from_pretrained(model_mapper[model_name])
    
#     # Load the dataset
#     dataset = load_dataset(dataset_mapper[data])
#     if data_subset:
#         dataset = dataset[data_split].select(range(data_subset))
#     else:
#         dataset = dataset[data_split]
#     test_loader = create_batches(dataset, batch_size)
    
#     results = []
#     pbar = tqdm(test_loader, total=len(dataset) // batch_size)
    
#     data_img_key = dataset_keys[data]['img']
#     data_text_key = dataset_keys[data]['text']
#     data_img_idx_key = dataset_keys[data]['img_idx']
    
#     model.eval() # Set the model to evaluation mode
    
#     for batch_idx, batch in enumerate(pbar):
#         predictions = []
        
#         # Get the images and captions for the batch
#         images_batch = batch[data_img_key]
#         captions_batch = batch[data_text_key]
#         images_names = batch[data_img_idx_key]
        
#         images_batch_cv2 = [pil_to_cv2(img) for img in images_batch]
        
#         with torch.no_grad():
#             inputs = processor(text=captions_batch, images=images_batch_cv2, return_tensors="pt", padding="max_length")
#             outputs = model(input_ids=inputs["input_ids"].to(DEVICE), pixel_values=inputs["pixel_values"].to(DEVICE))
        
#         # Get the cosine similarity scores
#         logits_per_image = outputs.logits_per_image # image text similarity
        
#         print(f'logits_per_image[0]: {logits_per_image[0]}')
#         print(f'len(logits_per_image): {len(logits_per_image)}')
        
#         # Compute the number of correct predictions
#         for i, cos_score in enumerate(logits_per_image):
#             curr_score = float(cos_score[0])
#             results.append({
#                 "image_name": images_names[i], 
#                 "data_name": dataset_mapper[data], 
#                 "model_name": model_mapper[model_name],
#                 "caption": captions_batch[i], 
#                 "cosine_score": curr_score
#             })
                
#         print(f'Avg Cosine Similarity Score so far: {np.mean([r["cosine_score"] for r in results])}')
        
#     results_df = pd.DataFrame(results)
    
#     accuracy = np.mean(results_df['cosine_score'])
#     print(f'Avg Cosine Similarity Score: {accuracy}')
    
#     os.makedirs(f'experiments/{eval_run_name}', exist_ok=True)
#     file_name = model_mapper[model_name].split('/')[-1]
#     results_df.to_csv(f'experiments/{eval_run_name}/{data}_cosine_{file_name}.csv')


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
    parser.add_argument('--use_finetuned', action='store_true', help='Use fine-tuned SemCLIP model')
    parser.add_argument('--verbose', action='store_true', help='Print model weights and shapes') # print semclip updated layers weights and shape to double check
    parser.add_argument('--cosine', action='store_true', help='Use direct cosine similarity for evaluation')
    parser.add_argument('--max_batch', action='store_true', help='Use the dataset length as the batch size')
    
    args = parser.parse_args()
    
    if not args.semclip:
        # if args.cosine:
        #     evaluate_clip_model_cosine(
        #         model_name=args.model_name,
        #         batch_size=args.batch_size,
        #         data=args.data,
        #         data_split=args.data_split,
        #         data_subset=args.data_subset,
        #         eval_run_name=args.eval_run_name,
        #     )
        evaluate_clip_model(
            model_name=args.model_name,
            batch_size=args.batch_size,
            data=args.data,
            data_split=args.data_split,
            data_subset=args.data_subset,
            eval_run_name=args.eval_run_name,
            max_batch=args.max_batch,
        )
    else:
        # if args.cosine:
        #     evaluate_semclip_model_cosine(
        #         model_name=args.model_name,
        #         pool_type=args.pool_type,
        #         projection_dim=args.projection_dim,
        #         multi_threading=args.multi_threading,
        #         text_pos_emb_2d=args.text_pos_emb_2d,
        #         data=args.data,
        #         batch_size=args.batch_size,
        #         data_split=args.data_split,
        #         data_subset=args.data_subset,
        #         eval_run_name=args.eval_run_name,
        #         use_finetuned=args.use_finetuned,
        #         verbose=args.verbose,
        #     )
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
            