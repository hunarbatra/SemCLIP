import argparse

from SemCLIP.semclip import get_segments_embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_name", type=str, help="image file name")
    parser.add_argument("--data_name", type=str, help="data directory name")
    args = parser.parse_args()
    get_segments_embeddings(args.image_name, args.data_name)