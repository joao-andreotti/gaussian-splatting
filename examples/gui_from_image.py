"""
This is an example on how to use the gui application to visualize your models.
"""


import os
import cv2
import math

from gaussian_splatting.model import GaussianCloud
from gaussian_splatting.gui import CudaRenderer, TkGUI
from gaussian_splatting.colmap import parse_cameras, parse_points3d,  parse_images, clean_text


dataset = "mug"

# reading the images from the image directory.
images = {
    image_name: cv2.imread(f"./data/{dataset}/images/{image_name}")[:, :, ::-1] / 255
    for image_name in os.listdir(f"./data/{dataset}/images/")
}
with open(f"./data/{dataset}/cameras.txt", "r")  as f:
    cameras = parse_cameras(clean_text(f.readlines()))
with open(f"./data/{dataset}/points3D.txt", "r")  as f:
    points3d = parse_points3d(clean_text(f.readlines()))
with open(f"./data/{dataset}/images.txt", "r")  as f:
    images = parse_images(clean_text(f.readlines()), cameras, points3d, images)



example_image = [*images.values()][0]



cloud = GaussianCloud.load(f"./models/{dataset}.pkl").eval()

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

