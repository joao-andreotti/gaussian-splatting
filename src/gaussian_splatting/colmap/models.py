__all__ = ["Camera", "Point3D", "Image"]

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy.typing as npt

@dataclass
class Camera:
    idx: int
    model: str # from colmap documentation
    width: int
    height: int
    parameters: list[str]

@dataclass
class Point3D:
    idx: int
    position: Tuple[float, float, float]
    color: Tuple[int, int, int]

@dataclass
class Image:
    idx: int
    view_position: Tuple[float, float, float] # translation
    view_orientation: Tuple[float, float, float, float] # quaternion
    camera: Camera
    name: str
    points: List[Tuple[float, float, Optional[Point3D]]]
    image: npt.NDArray # Image in [height, width, 3] format with RGB order channels