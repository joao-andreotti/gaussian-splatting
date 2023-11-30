__all__ = [
    "rotation_x",
    "rotation_y",
    "rotation_z",
    "normalize",
    "quaternion_to_rotation_matrix",
    "quaternions_to_rotation_tensors",
]

import numpy as np
import numpy.typing as npt
import torch


def rotation_x(theta: float) -> npt.NDArray[np.float64]:
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)],
    ])


def rotation_y(theta: float) -> npt.NDArray[np.float64]:
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)],
    ])


def rotation_z(theta: float) -> npt.NDArray[np.float64]:
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0,  0, 1],
    ])


def normalize(vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return vector if (norm:=np.linalg.norm(vector)) == 0 else vector / norm


def quaternion_to_rotation_matrix(quat: tuple[float, float, float, float]) -> npt.NDArray[np.float64]:
    """ Convert a tuple quaternion into a rotation matrix. """
    w, x, y, z = quat
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,          2*x*z + 2*y*w    ],
        [2*x*y + 2*z*w,         1 - 2*x*x - 2*z*z,      2*y*z - 2*x*w    ],
        [2*x*z - 2*y*w,         2*y*z + 2*x*w,          1 - 2*x*x - 2*y*y]
    ])

def quaternions_to_rotations_tensor(quaternions: torch.Tensor) -> torch.Tensor:
    """ Convert a tensor of quaternions into a tensor of rotation matrices.
    Args:
        quaternions (N, 4): tensor of quaternions.
    Returns:
        rotations (N, 3, 3): tensor of associated rotation matrices.
    """
    q0, q1, q2, q3 = quaternions.unbind(-1)
    rot_matrix = torch.stack([
        1 - 2*q2*q2 - 2*q3*q3, 2*q1*q2 - 2*q3*q0, 2*q1*q3 + 2*q2*q0,
        2*q1*q2 + 2*q3*q0, 1 - 2*q1*q1 - 2*q3*q3, 2*q2*q3 - 2*q1*q0,
        2*q1*q3 - 2*q2*q0, 2*q2*q3 + 2*q1*q0, 1 - 2*q1*q1 - 2*q2*q2
    ], dim=-1).reshape(-1, 3, 3)
    return rot_matrix