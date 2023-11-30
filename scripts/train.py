import os
import random
import argparse
from pathlib import Path

import cv2

from gaussian_splatting.colmap import parse_cameras, parse_images, parse_points3d, clean_text
from gaussian_splatting.model import View, GaussianCloud, train

random.seed(42)

def main(dataset_path: str, output_path: str, epochs: int):
    print(" ---> Loading dataset\n\n")

    base_path = Path(dataset_path)
    images = {
        image_name: cv2.imread(str(base_path / f"images/{image_name}"))[:, :, ::-1] / 255
        for image_name in os.listdir(base_path / "images")
    }
    with open(base_path / "cameras.txt", "r")  as f:
        cameras = parse_cameras(clean_text(f.readlines()))

    with open(base_path / "points3D.txt", "r")  as f:
        points3d = parse_points3d(clean_text(f.readlines()))

    with open(base_path / "images.txt", "r")  as f:
        images = parse_images(clean_text(f.readlines()), cameras, points3d, images)

    print(" ---> Preparing data\n\n")

    # creating views
    views = [View.from_image(image) for image in images.values()]

    # train/test splitting of views
    random.shuffle(views)
    train_split = 0.7
    dataset_size = len(views)
    train_dataset = views[:int(dataset_size*train_split)]
    test_dataset = views[int(dataset_size*train_split):]

    # creating the gaussian cloud
    gaussian_cloud = GaussianCloud.from_point_cloud([*points3d.values()]).to("cuda")

    # putting the parameters in train mode
    gaussian_cloud.train()


    print(" ---> Training the model \n\n")

    # training the model
    train(gaussian_cloud, train_dataset, test_dataset, epochs=epochs)


    print(" ---> Saving the model \n\n")

    # saving the model for later use
    gaussian_cloud.save(Path(output_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Gaussian Splatting.')
    parser.add_argument('dataset_path', type=str, help='Path to the desired dataset.')
    parser.add_argument('output_path', type=str, help='Output path of the pickled trained model.')
    parser.add_argument('--epochs', type=int, help='Number of training epochs', default=300)
    args = parser.parse_args() 

    main(args.dataset_path, args.output_path, args.epochs)