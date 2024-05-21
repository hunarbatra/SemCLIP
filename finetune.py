import os
import cv2
import torch
import wandb
import argparse

import torch.nn as nn
import numpy as np

from tqdm import tqdm
from PIL import Image
from datasets import load_dataset

from SemCLIP.semclip import SemCLIP
from SemCLIP.image_utils import DEVICE, create_batches, pil_to_cv2
from SemCLIP.model_utils import convert_models_to_fp32, convert_models_to_fp16
from config import dataset_mapper


def train_model(base_model='openai/clip-vit-base-patch32', pool_type='attention', projection_dim=512, data='COCO-13k', resume_training=False, train_name='semclip-v1', batch_size=64, num_epochs=3, lr=5e-5):
    # Initialize SemCLIP
    semclip = SemCLIP(model_name=base_model, pool_type='attention', projection_dim=512, device=DEVICE)
    
    # Load dataset
    dataset = load_dataset(dataset_mapper[data])
    
    wandb.login(key=os.getenv("WANDB_API_KEY"))

    semclip.model.to(DEVICE)

    learning_rate = lr # huggingface trainer default lr: 5e-5
    betas = (0.9, 0.999) # huggingface trainer default betas
    epsilon = 1e-6 # huggingface trainer default epsilon
    weight_decay = 0.2 # L2 regularization - finetuning CLIP with a small dataset can lead to overfitting so we add L2 regularization
    # try setting weight_decay to 0.001 (ref: https://github.com/openai/CLIP/issues/150#issuecomment-976076317)

    wandb_config = {
        "learning_rate": learning_rate,
        "betas": betas,
        "epsilon": epsilon,
        "weight_decay": weight_decay,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
    }

    if DEVICE == "cpu":
        semclip.model.float() # convert the model params to float if using CPU

    optimizer = torch.optim.AdamW(semclip.model.parameters(), lr=learning_rate, betas=betas, eps=epsilon, weight_decay=weight_decay)

    loss_fn = torch.nn.CrossEntropyLoss()

    train_loader = create_batches(dataset['train'], batch_size)

    checkpoint_dir = "model_ckpts"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{train_name}_checkpoint.pth")

    start_epoch = 0
    start_batch = 0
    wandb_run_id = None

    if resume_training and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        semclip.model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        start_batch = checkpoint['batch'] + 1
        wandb_run_id = checkpoint['wandb_run_id']
        print(f"Resuming training from epoch {start_epoch}, batch {start_batch}")
        
    if wandb_run_id:
        wandb.init(project="semclip", name=train_name, id=wandb_run_id, resume="must", config=wandb_config)
    else:
        wandb_run = wandb.init(project="semclip", name=train_name, config=wandb_config)
        wandb_run_id = wandb_run.id

    for epoch in range(start_epoch, num_epochs):
        semclip.model.train()
        
        pbar = tqdm(train_loader, total=len(dataset['train']) // batch_size, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch_idx, batch in enumerate(pbar):
            if epoch == start_epoch and batch_idx < start_batch:
                continue  # Skip batches already processed in the current epoch if resuming
            
            optimizer.zero_grad()
            
            image_batch_pil = batch['downloaded_img']
            text_batch = batch['caption']
            
            # Convert the batch of PIL images to OpenCV images
            image_batch_cv2 = [pil_to_cv2(img) for img in image_batch_pil]
            
            if DEVICE != "cpu":
                convert_models_to_fp32(semclip.model)

            # Forward pass through the model
            try:
                image_embeddings, text_embeddings = semclip.get_semclip_embeddings_direct_img(images=image_batch_cv2, captions=text_batch)
            except Exception as e:
                print(f"error: {e}; images batch being processed: {batch['image']}")
                continue
            
            # Process final embeddings (normalize, compute logits)
            logits_per_image, logits_per_text = semclip.process_final_embeddings(image_embeddings, text_embeddings)
                
            # Compute the loss
            ground_truth = torch.arange(len(logits_per_text)).to(DEVICE)
            text_loss = loss_fn(logits_per_text, ground_truth) # contrastive loss
            image_loss = loss_fn(logits_per_text.t(), ground_truth) # contrastive loss
            total_loss = (text_loss + image_loss) / 2.0
                
            # Backward pass
            total_loss.backward()
            
            # if the device is CPU, directly update the model
            if DEVICE == "cpu":
                optimizer.step()
            else:
                convert_models_to_fp32(semclip.model)
                optimizer.step()
                convert_models_to_fp16(semclip.model)
            
            # Save checkpoint after each batch
            print(f'Saving checkpoint at... {checkpoint_path}')
            torch.save({
                'epoch': epoch,
                'batch': batch_idx,
                'model_state_dict': semclip.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'wandb_run_id': wandb_run_id,
            }, checkpoint_path)
            
            # Log the loss to wandb after each batch
            wandb.log({"Loss": total_loss.item()})
            
            # Update the progress bar with the current loss
            pbar.set_postfix(Loss=total_loss.item())

    semclip.upload_model_to_hf_hub(model_name=train_name, hf_name='hunarbatra')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SemCLIP model")
    parser.add_argument('--data', type=str, default='COCO-13k', help='Dataset name, default: COCO-13k - check config.py for more dataset options or to add more datasets')
    parser.add_argument('--resume_training', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--train_name', type=str, default='semclip-v1', help='Training name for checkpointing and wandb')
    parser.add_argument('--base_model', type=str, default='openai/clip-vit-base-patch32', help='Base model for SemCLIP')
    parser.add_argument('--pool_type', type=str, default='attention', help='Pooling type for SemCLIP; options: "mean", "cls", "attention"')
    parser.add_argument('--projection_dim', type=int, default=512, help='Projection dimension for SemCLIP')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs for training') # note: huggingface trainer default num_train_epochs = 3
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate for training')
    
    args = parser.parse_args()
    
    train_model(
        base_model=args.base_model, 
        pool_type=args.pool_type,
        projection_dim=args.projection_dim,
        data=args.data,
        resume_training=args.resume_training,
        train_name=args.train_name,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
    )
