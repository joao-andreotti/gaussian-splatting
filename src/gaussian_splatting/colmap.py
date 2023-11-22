from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class Camera:
    idx: int
    model: str
    width: int
    height: int
    parameters: list

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
    ) -> Dict[int, Image]:
    images: Dict[int, Image] = {}
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
            points=points
        )

        images[image.idx] = image

    return images

if __name__ == "__main__":
    with open("cloud/cameras.txt", "r")  as f:
        cameras = parse_cameras(clean_text(f.readlines()))

    with open("cloud/points3D.txt", "r")  as f:
        points3d = parse_points3d(clean_text(f.readlines()))

    with open("cloud/images.txt", "r")  as f:
        images = parse_images(clean_text(f.readlines()), cameras, points3d)