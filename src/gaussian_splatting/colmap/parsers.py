__all__  = ["clean_text", "parse_cameras", "parse_points3d", "parse_images"]

from typing import Dict, List, Tuple

import numpy.typing as npt

from gaussian_splatting.colmap.models import Camera, Image, Point3D

def gather_by(step: int, data: list) -> List[Tuple]:
    lists = [data[start::step] for start in range(step)]
    return [*zip(*lists)]

def remove_comments(source: List[str]) -> List[str]:
    return [line for line in source if not line.startswith("#")]

def clean_text(source: List[str]) -> List[str]:
    without_comments = remove_comments(source)
    return [line.rstrip("\n") for line in without_comments]

def parse_cameras(source: List[str]) -> Dict[int, Camera]:
    cameras: Dict[int, Camera] = {}
    for line in source: 
        [idx, model, width, height, *parameters] = line.split(" ")
        cameras[int(idx)] = Camera(
            idx=int(idx),
            model=model,
            width=int(width),
            height=int(height),
            parameters=[param for param in parameters]
        )
    return cameras

def parse_points3d(source: List[str]) -> Dict[int, Point3D]:
    points3d: Dict[int, Point3D] = {}
    for line in source:
        [point3d_id, x, y, z, r, g, b, *_] = line.split(" ")
        position = (float(x), float(y), float(z))
        color = (int(r), int(g), int(b))
        point = Point3D(int(point3d_id), position, color)
        points3d[point.idx] = point
    return points3d

def parse_images(
        source: List[str], 
        cameras: Dict[int, Camera], 
        points3d: Dict[int, Point3D],
        images: Dict[str, npt.NDArray]
    ) -> Dict[int, Image]:
    """Parse colmap image information.
    Args:
        source: the source content of the colmap file.
        cameras: related cameras indexed by their id,
        points3d: related 3d points from colmap indexed by their id,
        images: loaded images indexed by their name e.g. "0019.png"
    Returns:
        Images indexed by their colmap index.
    """
    parsed_images: Dict[int, Image] = {}
    image_lines = gather_by(2, source) # grabing in pairs of lines
    for image_data, points_data in image_lines:
        [idx, qw, qx, qy, qz, tx, ty, tz, camera_id, name, *rest] = image_data.split(" ")

        if rest: 
            raise Exception("bad format. Too much info on line for image data")

        if not int(camera_id) in cameras.keys():
            raise Exception("camera not found")

        camera = cameras[int(camera_id)]
        quaternion = (float(qw), float(qx), float(qy), float(qz))
        position = (float(tx), float(ty), float(tz))
        points = [
            (float(x), float(y), points3d.get(int(point_3d_id)) if point_3d_id != "-1" else None)
            for x, y, point_3d_id in gather_by(3, points_data.split(" "))
        ]
        
        image = Image(
            idx=int(idx),
            view_position=position,
            view_orientation=quaternion,
            camera=camera,
            name=name,
            points=points,
            image=images[name],
        )

        parsed_images[image.idx] = image

    return parsed_images