import math
import numpy as np

from gaussian_splatting.gui.renderer import Renderer
from gaussian_splatting.model.view import View
from gaussian_splatting.model.gaussian_cloud import GaussianCloud
from gaussian_splatting.model.util import create_rasterizer


class CudaRenderer(Renderer):
    def __init__(
        self,
        gaussian_cloud: GaussianCloud
    ) -> None:
        """Inintializes the cuda rasterizer with a given dataset.
        
        N is the number of gaussians in the given dataset.
        All the tensors will be casted on a CUDA device.

        Args:
            means3D (float tensor N x 3): center of each gaussian
            shs (float tensor N x 16 x 3): spheric harmonics for the rgb colors of each gaussian.
            opacities (float tensor N x 1): opacity value for each gaussian.
            scales (float tensor N x 3): scale of the x, y and z values for the variance of the gaussian.
                In other words, diagonal of the covariance matrix for the gaussian if it is centered and
                axis-aligned.
            rotations (float tensor N x 4): quaternion representing the rotation from the center and 
                axis-aligned gaussian to its actual orientation in space.

        Raises:
            Exception if the cuda device is not available  
        """
        self.gaussian_cloud = gaussian_cloud.to("cuda")

    def render(
        self,
        width: int,
        height: int,
        position: tuple[float, float, float],
        world_z: tuple[float, float, float],
        world_x: tuple[float, float, float],
        *,
        fov: tuple[float, float] = (math.pi/3, math.pi/3),
        background_color: tuple[float, float, float] = (0, 0, 0),
    ) -> np.ndarray:
        fovx, fovy = fov
        x = np.array(world_x)
        z = np.array(world_z)
        y = np.cross(z, x)
        rotation = np.block([[x], [y], [z]]).T
        view = View(
            position=position,
            rotation=rotation,
            height=height,
            width=width,
            fovx=fovx,
            fovy=fovy,
            image=None
        )
        rasterizer = create_rasterizer(view, background_color=background_color)
        rasterized_image, _ = rasterizer(**self.gaussian_cloud.parameters)
        return rasterized_image.cpu().detach().numpy().transpose([1, 2, 0])