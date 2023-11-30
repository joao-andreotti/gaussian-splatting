__all__ = ["GaussianCloud"]

from typing import Literal
import pickle

import torch

from gaussian_splatting.colmap.models import Point3D

class GaussianCloud:
    def __init__(
        self, 
        means3D: torch.Tensor,
        means2D: torch.Tensor,
        shs: torch.Tensor,
        opacities: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
    ) -> None:
        """Inintializes the gaussian Cloud with the given parameters.
        
        N is the number of gaussians in the given dataset.
        By default tensors will be kept on its device.

        Args:
            means3D (float tensor N x 3): center of each gaussian
            means2D (float tensor N x 3): center of each gaussian in view-space (after projecting on camera)
                These are used mainly for calculating the gradients.
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
        self.means3D = means3D
        self.means2D = means2D
        self.shs = shs
        self.opacities = opacities
        self.scales = scales
        self.rotations = rotations

    def to(self, device: Literal["cpu", "cuda"]) -> "GaussianCloud":
        """Cast all parameters of the cloud to desired device.
        Args: 
            device: target device of casting
        Raises:
            Exception: if the device is not available.
        Returns:
            This same gaussian cloud with parameters in desired device.
        """
        if device == "cuda" and not torch.cuda.is_available():
            raise Exception("Cuda device is not available")

        self.means3D = self.means3D.to(device)
        self.means2D = self.means2D.to(device)
        self.shs = self.shs.to(device)
        self.opacities = self.opacities.to(device)
        self.scales = self.scales.to(device)
        self.rotations = self.rotations.to(device)
        return self
    
    def center(self) -> "GaussianCloud":
        """Center the cloud on the world position space.
        Returns:
            self
        """
        self.means3D -= self.means3D.mean(axis=0)
        return self

    def __repr__(self) -> str:
        return f"Gaussian Cloud of {self.means3D.shape[0]} points in {self.means3D.device}"

    @property
    def parameters(self) -> dict[str, torch.Tensor]:
        """Dictionary with all the intrinsic model parameters.
        Tensors are in cuda device by default.
        """
        return dict(
            means3D=self.means3D,
            means2D=self.means2D,
            shs=self.shs,
            opacities=self.opacities,
            scales=self.scales,
            rotations=self.rotations,
        )

    def save(self, path: str) -> None:
        """Pickles the cloud in a file.

        Tensors will be stored in the CPU device for compatilibity.

        Args:
            path: path in which to save the pickled cloud.
        """
        with open(path, "wb") as file:
            pickle.dump(self.parameters, file)

    @classmethod
    def load(cls, path: str) -> "GaussianCloud":
        """Loads pickled cloud from a file.

        Args:
            path: path from which to load the pickled cloud.
        """
        with open(path, "rb") as file:
            return cls(**pickle.load(file))
        
    def train(self) -> "GaussianCloud":
        """Puts all parameters in training mode.
        This makes all tensors require a gradient.
        """
        self.means3D.requires_grad = True
        self.means2D.requires_grad = True
        self.shs.requires_grad = True
        self.opacities.requires_grad = True
        self.scales.requires_grad = True
        self.rotations.requires_grad = True
        return self
    
    def eval(self) -> "GaussianCloud":
        """Puts all parameters in evaluation mode.
        This makes all tensors not require a gradient.
        This leads to faster computations.
        """
        self.means3D.requires_grad = False
        self.means2D.requires_grad = False
        self.shs.requires_grad = False
        self.opacities.requires_grad = False
        self.scales.requires_grad = False
        self.rotations.requires_grad = False
        return self
    
    @classmethod
    def from_point_cloud(
        cls, 
        cloud: list[Point3D],
        *,
        scale: tuple[float, float, float] = (.01, .01, .01),
    ) -> "GaussianCloud":
        """Creates a Gaussian Cloud directly from a point cloud.
        Uses default parameters passed in.

        Args:
            cloud: the point cloud to be considered.
            scale: defaul scale for each gaussian on initialization.
        Returns:
            initialized gaussian cloud.
        """
        C0 = 0.28209479177387814
        number_points = len(cloud)
        rgb_colors = torch.tensor([
            [list(point.color), * [[0, 0, 0]]*15]
            for point in cloud
        ]) / 256 
        sh_colors = (rgb_colors - 0.5) / C0
        return cls(
            means3D=torch.tensor([point.position for point in cloud], dtype=torch.float),
            means2D=torch.zeros((number_points, 3), dtype=torch.float),
            shs=sh_colors.float(),
            opacities=torch.ones([number_points, 1], dtype=torch.float),
            scales=torch.tensor([list(scale)] * number_points, dtype=torch.float),
            rotations=torch.tensor([[1, 0, 0, 0]] * number_points, dtype=torch.float),
        )