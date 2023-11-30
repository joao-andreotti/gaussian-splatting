__all__ = ["View"]

from typing import Tuple, Optional

import numpy as np
import numpy.typing as npt

from gaussian_splatting.colmap import Image
from gaussian_splatting.utils import quaternion_to_rotation_matrix

class View: 
    def __init__(
        self, 
        position: tuple[float, float, float],
        rotation: np.ndarray,   # 4x4 rotation matrix for the camera
                                # starting from x=left y=top z=front
        height: int,
        width: int,
        fovx: float,
        fovy: float,
        image: Optional[npt.NDArray] = None
    ) -> None:
        self.image = image
        self.position = np.array(position)
        self.rotation: np.ndarray = rotation
        self.width = width
        self.height = height
        self.fovx = fovx
        self.fovy = fovy

    @classmethod
    def from_orientation(
        cls, 
        position: tuple[float, float, float],
        orientation: tuple[float, float, float, float],
        height: int,
        width: int,
        fovx: float,
        fovy: float,
        image: Optional[npt.NDArray] = None
    ) -> "View":
        return cls(
            image=image,
            position=position,
            rotation=quaternion_to_rotation_matrix(orientation),
            width=width,
            height=height,
            fovx=fovx,
            fovy=fovy,
        )
    
    def __repr__(self) -> str:
        return f"View at {tuple(self.position)}"

    @classmethod
    def from_image(cls, image: Image) -> "View":
        if image.camera.model == "PINHOLE":
            [focalx, focaly, *_] = image.camera.parameters
            focal: Tuple[float, float] = float(focalx), float(focaly)
            
        elif image.camera.model == "SIMPLE_RADIAL":
            [focal, *_] = image.camera.parameters
            focal = float(focal), float(focal)
        else:
            raise Exception(f"Camera model {image.camera.model} not allowed.")
        focalx, focaly = focal

        return cls.from_orientation(
            image=image.image,
            position=image.view_position,
            orientation=image.view_orientation,
            width=image.camera.width,
            height=image.camera.height,
            fovx = 2 * np.arctan(image.camera.width /(2*focalx)),
            fovy = 2 * np.arctan(image.camera.height/(2*focaly))
        )
    
    def viewmatrix(self, include_intrinsics: bool = False):
        viewmatrix = np.block([
            [self.rotation,     self.position.reshape(-1, 1) ],
            [np.zeros((1, 3)),  1   ]
        ])

        if not include_intrinsics:
            return viewmatrix
        
        fx, fy = self.focal
        cx, cy = self.principal
        intrinsics = np.array([
            [fx,    0,      cx,     0],
            [0,     fy,     cy,     0],
            [0,     0,      1,      0],
            [0,     0,      0,      1]
        ])
        return intrinsics @ viewmatrix

    def projmatrix(
        self, 
        znear: float = 0.1,
        zfar: float = 100,
    ) -> np.ndarray:
        tanfovx = np.tan(self.fovx/2)
        tanfovy = np.tan(self.fovy/2)
        top = znear * tanfovy
        bottom = -top
        right = znear * tanfovx
        left = - right
        return np.array([
            [2 * znear / (right - left),    0,                          (right+left)/(right-left),      0                      ],
            [0,                             2 * znear / (top - bottom), (top + bottom)/(top - bottom),  0                      ],
            [0,                             0,                          (znear + zfar)/(zfar - znear),  zfar*znear/(zfar-znear)],
            [0,                             0,                          1,                              0                      ]
        ])
