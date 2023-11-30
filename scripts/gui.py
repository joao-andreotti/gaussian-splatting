import math
import os
import cv2
from pathlib import Path
import random
import argparse

from gaussian_splatting.model import GaussianCloud
from gaussian_splatting.gui import CudaRenderer, TkGUI
from gaussian_splatting.colmap import parse_cameras, parse_points3d,  parse_images, clean_text


def gui_from_center(model_path: str):

    cloud = GaussianCloud.load(model_path).eval().center()

    renderer = CudaRenderer(cloud)

    gui = TkGUI(
        renderer=renderer, 
        height=1024, 
        width=1024, 
        initial_position=(0, 0, 0),
        background_color=(0, 0, 0),
        fov=(math.pi/3, math.pi/3),
        help_on_startup=False
    )

    gui.start()

def gui_from_view(model_path: str, dataset_path: str):

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

    example_image = random.choice([*images.values()])

    cloud = GaussianCloud.load(model_path).eval()

    renderer = CudaRenderer(cloud)

    gui = TkGUI(
        renderer=renderer, 
        height=1024, 
        width=1024, 
        initial_position=(1, 1, 1),
        background_color=(0, 0, 0),
        fov=(math.pi/3, math.pi/3),
        help_on_startup=False
    ).view_from_image(example_image)

    gui.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Gaussian Splatting.')
    parser.add_argument('model', type=str, help='Path to the pickle of the desired model (example ./models/cat.pkl)')
    parser.add_argument('--dataset', type=str, help='Path to the corresponding dataset. If not provided camera will start from center.', default=None)
    args = parser.parse_args() 
    if args.dataset is not None:
        gui_from_view(
            model_path=args.model,
            dataset_path=args.dataset
        )
    else:
        gui_from_center(model_path=args.model_path)