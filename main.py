import argparse

from SemCLIP.semclip import SemCLIP


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_name", type=str, help="image file name")
    parser.add_argument("--data_name", type=str, help="data directory name")
    parser.add_argument("pool_type", type=str, default="mean", help="pooling type; options: ['mean', 'cls', 'attention'], default: mean")
    parser.add_argument("--projection_dim", type=int, default=None, help="custom projection dimension, default: 512")
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32", help="CLIP model name, default: openai/clip-vit-base-patch32")
    args = parser.parse_args()
    
    semclip = SemCLIP(model_name=args.model_name, device=DEVICE)
    semclip.get_segments_embeddings(args.image_name, args.data_name, args.projection_dim)