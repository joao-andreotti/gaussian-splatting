import argparse
import pathlib
import gdown
import os

# drive link to the nerf synthetic dataset
def download(path: str):
    path = pathlib.Path(path)
    url = "https://drive.google.com/uc?id=18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG"
    output = path / "dataset.zip"
    gdown.cached_download(url, output, postprocess=gdown.extractall)
    os.remove(path / "dataset.zip")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download nerf synthetic dataset')
    parser.add_argument('--path', type=str, help='Path in which to save the dataset', default="./data/")
    args = parser.parse_args() 
    download(args.path)