"""
This is an example on how to use the gui application to visualize your models.
"""

from gaussian_splatting.model import GaussianCloud
from gaussian_splatting.gui import CudaRenderer, TkGUI

# change for the pickle path of your model.
model = "./models/hotdog.pkl"

cloud = GaussianCloud.load(model).eval().center()

renderer = CudaRenderer(cloud)

gui = TkGUI(
    renderer=renderer, 
    height=1024, 
    width=1024, 
    initial_position=(1, 1, 1),
    background_color=(1, 1, 1),
    help_on_startup=True
)

gui.start()

