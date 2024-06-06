import argparse

from SemCLIP.semclip import SemCLIP
from SemCLIP.image_utils import DEVICE


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_name", type=str, help="image file name")
    parser.add_argument("--data_name", type=str, help="data directory name")
    parser.add_argument("--pool_type", type=str, default="attention", help="pooling type; options: ['mean', 'cls', 'attention'], default: mean")
    parser.add_argument("--projection_dim", type=int, default=None, help="custom projection dimension, default: 512")
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32", help="CLIP model name, default: openai/clip-vit-base-patch32")
    parser.add_argument("--text", type=str, default="a photo of a dog", help="text input")
    parser.add_argument("--text_pos_emb_2d", action="store_true", help="Use 2D positional embeddings for text")
    args = parser.parse_args()
    
    semclip = SemCLIP(
        model_name=args.model_name, 
        pool_type=args.pool_type, 
        projection_dim=args.projection_dim, 
        device=DEVICE,
        text_pos_emb_2d=args.text_pos_emb_2d,
    )
    
    image_features = semclip.get_image_features(args.image_name, args.data_name, return_embeds=True)
    text_features = semclip.get_text_features(args.text, return_embeds=True)
    
    logits_per_image, logits_per_text = semclip(images=image_features, texts=text_features, raw_embeds=True)
    
    print(f'semclip logits_per_image: {logits_per_image}')
    print(f'semclip logits_per_text: {logits_per_text}')
