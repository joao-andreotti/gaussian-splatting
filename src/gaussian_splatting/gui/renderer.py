__all__ = ["Renderer"]

import math
from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt

class Renderer(ABC):
    @abstractmethod
    def render(
        self,
        width: int,
        height: int,
        position: npt.NDArray[np.float64],
        world_z: npt.NDArray[np.float64],
        world_x: npt.NDArray[np.float64],
        *,
        fov: tuple[float, float] = (math.pi/3, math.pi/3),
        background_color: tuple[float, float, float] = (0, 0, 0),
    ) -> np.ndarray:
        """Render the model from the perspective provided.

        The vector arguments provided are all in the camera perspective/basis.

        World y axis is induced from the z and x axis passed in.

        Args:
            width: width of the returning image in pixels.
            height: height of the returning image in pixels.
            position (3,): position of the world center from the camera perspective.
            world_z (3,): world z axis from camera perspective.
                Must be unitary vector.
            world_x (3,): world x axis from camera perspective.
                Must be unitary vector and perpendicular to world_z.

            fov: camera field of view in radians for x and y image axis.
                Must be between 0 and pi.
                Defaults to pi / 3 in both directions.
            background_color: background color for the rendering in (R, G, B).
                Each channel assumes values between 0 and 1.
                Defaults to black.

        Returns:
            Rendered image in float numpy array format with shape (height, width, channels).
            Channels=3 for Red, Green and Blue channels in this order.
            Values for the colors are between 0 and 1.
        """
        ...
