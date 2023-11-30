"""
This is an example on how to use the gui application to visualize your models.
"""

import math

from gaussian_splatting.model import GaussianCloud
from gaussian_splatting.gui import CudaRenderer, TkGUI


dataset = "mug"

cloud = GaussianCloud.load(f"./models/{dataset}.pkl").eval().center()

renderer = CudaRenderer(cloud)

gui = TkGUI(
    renderer=renderer, 
    height=1024, 
    width=1024, 
    initial_position=(1, 1, 1),
    background_color=(0, 0, 0),
    fov=(math.pi/3, math.pi/3),
    help_on_startup=False
)

gui.start()

