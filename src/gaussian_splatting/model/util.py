__all__ = ["create_rasterizer", "position_points"]

import torch
import numpy as np

import diff_gaussian_rasterization as dg

from gaussian_splatting.model.view import View
from gaussian_splatting.utils import quaternions_to_rotations_tensor


def position_points(
    scales: torch.Tensor,
    rotations: torch.Tensor,
    points: torch.Tensor
) -> torch.Tensor:
    """Position a set of points by applying scaling and rotation.
    Make sure the tensors are on the same device.
    N is the number of points on which to apply the transformations.
    Args:
        scales (N, 3): scaling factor of the point from the origin.
        rotations (N, 4): rotation to be performed from the origin in quaternion format.
        points (N, 3): points to be considered.
    Returns:
        new points. Now scaled from the origin and then rotated.
    """
    scaled = scales * points
    quaternions = rotations / rotations.norm(dim=1, keepdim=True)
    rot_matrices = quaternions_to_rotations_tensor(quaternions)
    rotated_points = torch.bmm(rot_matrices, scaled.unsqueeze(-1)).squeeze(-1)
    return rotated_points

def create_rasterizer(view: View, *, background_color=(0, 0, 0)):
    configuration = dg.GaussianRasterizationSettings(
        image_height=view.height,
        image_width=view.width,
        tanfovx=np.tan(view.fovx/2),
        tanfovy=np.tan(view.fovy/2),
        bg=torch.tensor(list(background_color), dtype=torch.float, device="cuda"),
        scale_modifier=1.,
        viewmatrix=torch.tensor(view.viewmatrix(False).T, dtype=torch.float, device="cuda"),
        projmatrix=torch.tensor(view.viewmatrix(False).T @ view.projmatrix(znear=1).T, dtype=torch.float, device="cuda"),
        sh_degree=0,
        campos=torch.tensor(view.position, dtype=torch.float, device="cuda"),
        prefiltered=False,
        debug=False
    )
    return dg.GaussianRasterizer(configuration)