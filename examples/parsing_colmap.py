"""
This is an example script on how to parse colmap files using this library.
"""
import os
import cv2

from gaussian_splatting.colmap import parse_cameras, parse_points3d,  parse_images, clean_text

# reading the images from the image directory.
images = {
    image_name: cv2.imread(f"images/{image_name}")[:, :, ::-1] / 255
    for image_name in os.listdir(f"images")
}

# parsing all the colmap data into clean datastructures
with open(f"cameras.txt", "r")  as f:
    cameras = parse_cameras(clean_text(f.readlines()))

with open(f"points3D.txt", "r")  as f:
    points3d = parse_points3d(clean_text(f.readlines()))

with open(f"images.txt", "r")  as f:
    images = parse_images(clean_text(f.readlines()), cameras, points3d, images)