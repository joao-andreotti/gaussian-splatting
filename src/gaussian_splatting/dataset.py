from dataclasses import dataclass
from typing import List, Iterable, Dict, Tuple
from gaussian_splatting.colmap import Camera, Point3D, Image

import numpy as np

C0 = 0.28209479177387814

def normalize(vector: np.ndarray) -> np.ndarray:
    return vector if (norm:=np.linalg.norm(vector)) == 0 else vector / norm

def quaternion_to_rotation_matrix(quat: Tuple[float, float, float, float]):
    """ Convert a quaternion into a rotation matrix. """
    w, x, y, z = quat
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,          2*x*z + 2*y*w    ],
        [2*x*y + 2*z*w,         1 - 2*x*x - 2*z*z,      2*y*z - 2*x*w    ],
        [2*x*z - 2*y*w,         2*y*z + 2*x*w,          1 - 2*x*x - 2*y*y]
    ])

class Cloud:
    def __init__(self, points: Iterable[Point3D]):
        self.points: List[Point3D] = [*points]

    def points_positions(self):
        return np.array([
            point.position
            for point in self.points
        ])
    
    def points_colors(self):
        rgb_colors =  np.array([
            [list(point.color), * [[0, 0, 0]]*15]
            for point in self.points
        ]) / 256
        return (rgb_colors - 0.5) / C0

class View: 
    def __init__(self, image: Image) -> None:
        self.image = image
        
        # position of the world center from this view/camera
        self.position: np.ndarray = np.array(image.view_position)

        # rotation form the world to the camera
        self.rotation: np.ndarray = quaternion_to_rotation_matrix(image.view_orientation)

        # image constants
        self.height: int = self.image.camera.height
        self.width: int = self.image.camera.width

        # intrinsics of the camera
        if self.image.camera.model == "PINHOLE":
            [focalx, focaly, principalx, principaly] = self.image.camera.parameters
            self.focal: Tuple[float, float] = float(focalx), float(focaly)
            self.principal: Tuple[float, float] = float(principalx), float(principaly)
            
        elif self.image.camera.model == "SIMPLE_RADIAL":
            [focal, principalx, principaly, *_] = self.image.camera.parameters
            self.focal = float(focal), float(focal)
            self.principal: Tuple[float, float] = float(principalx), float(principaly)
        else:
            raise Exception(f"Camera model {self.image.camera.model} not allowed.")
        
        # fov
        self.fovx = 2 * np.arctan(self.width/(2*self.focal[0]))
        self.fovy = 2 * np.arctan(self.height/(2*self.focal[1]))
    
    def viewmatrix(self, include_intrinsics: bool = False):
        eye = - self.rotation.T @ self.position.reshape(-1, 1)
        viewmatrix = np.block([
            [self.rotation,     eye ],
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
        fovx: float = np.pi / 2, 
        fovy: float = np.pi / 2 
    ) -> np.ndarray:
        tanfovx = np.tan(fovx/2)
        tanfovy = np.tan(fovy/2)
        top = znear * tanfovy
        bottom = -top
        right = znear * tanfovx
        left = - right
        return np.array([
            [2 * znear / (right - left),    0,                          (right+left)/(right-left),      0                        ],
            [0,                             2 * znear / (top - bottom), (top + bottom)/(top - bottom),  0                        ],
            [0,                             0,                          (znear + zfar)/(zfar - znear),  2*zfar*znear/(zfar-znear)],
            [0,                             0,                          1,                              0                        ]
        ])