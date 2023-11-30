__all__ = ["TkGUI"]

from typing import Optional

import tkinter as tk
import numpy as np
from PIL import Image as PILImage, ImageTk

from gaussian_splatting.model.view import Image
from gaussian_splatting.gui.renderer import Renderer
from gaussian_splatting.gui.gui import GUI
from gaussian_splatting.utils import quaternion_to_rotation_matrix, rotation_x, rotation_y, rotation_z


HELP_TEXT = f"""

Esc:                    Close the window.
H:                      Open this help window.

Mouse controls:
    Left button drag:   Control the Yaw and Pitch.
    Mid button drag:    Control the displacement.
    Right button drag:  Control the Roll of the image (only y axis movement).
    Scroll:             Control the closeness to the camera / zoom.
    

keyboard controls:
    A, D:               Control the y displacement.
    W, S:               Control the closeness to the camera / zoom.
    Z, Space:           Control the x displacement.
    Up, Down:           Control the Pitch.
    Left, Right:        Control the Yaw.
    E, Q:               Control the Roll (only y axis movement).

"""

def reactive(update):
    """Decorator for TK listener methods that triggers updates.
    Decorate any listening before binding to call the update funciton after. 
    Args:
        update ((self) -> None): class method that updates the UI accordingly.
            It receives no arguments and returns None.
    """
    def decorator(method):
        def wrapper(self, event: tk.Event):
            method(self, event)
            update(self)
        return wrapper
    return decorator

class TkGUI(GUI):
    def __init__(
        self, 
        renderer: Renderer,
        *,
        width: int,
        height: int,
        initial_position: tuple[float, float, float] = (0, 0, 0),
        background_color: tuple[float, float, float] = (0, 0, 0),
        fov: tuple[float, float] = (0, 0),
        help_on_startup: bool = True,
    ) -> None:
        self.renderer = renderer    
        self.width = width
        self.height = height
        self.running = False
        self.background_color = background_color
        self.fov = fov
        self.help_on_startup = help_on_startup

        # initialize default camera parameters
        self.world_position = np.array(initial_position, dtype=np.float64)
        self.world_z = np.array([0, 0, 1], dtype=np.float64)
        self.world_x = np.array([1, 0, 0], dtype=np.float64)

        # initialize event listener parameters
        self.mouse_x: Optional[int] = None
        self.mouse_y: Optional[int] = None


    def view_from_image(self, image: Image) -> "TkGUI":
        orientation = image.view_orientation
        rotation = quaternion_to_rotation_matrix(orientation)
        self.world_position = np.array(image.view_position, dtype=np.float64)
        self.world_x = rotation[:, 0].astype(np.float64)
        self.world_z = rotation[:, 2].astype(np.float64)
        return self

    def start(self) -> None:
        """Starts the GUI application.
        """
        self.label: Optional[tk.Label] = None
        self.root = tk.Tk()
        self.root.title("Gaussian Splatting")
        self.root.bind('<Key>', self._on_press_key)
        self.root.bind("<Button-4>", self._on_scroll_up)
        self.root.bind("<Button-5>", self._on_scroll_down)
        self.root.bind("<B1-Motion>", self._on_mouse_left_drag)
        self.root.bind("<B2-Motion>", self._on_mouse_middle_drag)
        self.root.bind("<B3-Motion>", self._on_mouse_right_drag)
        self.root.bind("<ButtonRelease-1>", self._release_mouse)
        self.root.bind("<ButtonRelease-2>", self._release_mouse)
        self.root.bind("<ButtonRelease-3>", self._release_mouse)


        image = self._get_image()
        self.label = tk.Label(self.root, image=image)
        self.label.pack()
        self.running = True

        if self.help_on_startup: self._show_instructions()

        self.root.mainloop()

    def _get_image(self) -> ImageTk.PhotoImage:
        rendered = self.renderer.render(
            width=self.width,
            height=self.height,
            position=self.world_position,
            world_z=self.world_z,
            world_x=self.world_x,
            background_color=self.background_color,
            fov=self.fov
        )
        new_image = (rendered * 255).astype(np.uint8)
        return ImageTk.PhotoImage(PILImage.fromarray(new_image))
    
    def _update_image(self):
        if self.label is None:
            raise Exception("Label not yet available. Check if application has started.")
        new_photo = self._get_image()
        self.label.config(image=new_photo)
        self.label.image = new_photo
        if not self.running:
            self.root.destroy()

    def _show_instructions(self):
        popup = tk.Toplevel(self.root)
        popup.attributes("-topmost", True)
        popup.title("Instructions")
        tk.Label(popup, text="Welcome to the Gaussian Splatting GUI", font=("Courier", 16)).pack(pady=10, padx=10, fill='x')
        tk.Label(popup, justify='left', anchor='w', font=("Courier", 12), text=HELP_TEXT).pack(pady=10, padx=10, fill='x')

    # TK event listeners
    @reactive(_update_image)
    def _on_mouse_left_drag(self, event: tk.Event):
        if (drag := self._drag_mouse(event)) is None:
            return
        dx, dy = drag
        step = 1e-3
        rotation = rotation_x(step*dy) @ rotation_y(-step*dx)
        self.world_z = (rotation @ self.world_z).astype(np.float64)
        self.world_x = (rotation @ self.world_x).astype(np.float64)

    @reactive(_update_image)
    def _on_mouse_middle_drag(self, event: tk.Event):
        if (drag := self._drag_mouse(event)) is None:
            return
        dx, dy = drag 
        step = 1e-3
        self.world_position += (
            step * dy * np.array([0, 1, 0]) + 
            step * dx * np.array([1, 0, 0])
        ).astype(np.float64)

    @reactive(_update_image)
    def _on_mouse_right_drag(self, event: tk.Event):
        if (drag := self._drag_mouse(event)) is None:
            return
        _, dy = drag
        step = 1e-3
        rotation = rotation_z(step * dy)
        self.world_position = (rotation @ self.world_position).astype(np.float64)
        self.world_z = (rotation @ self.world_z).astype(np.float64)
        self.world_x = (rotation @ self.world_x).astype(np.float64)
        self._update_image()

    @reactive(_update_image)
    def _on_scroll_down(self, event: tk.Event):
        step = .1
        self.world_position += step * np.array([0, 0, 1])

    @reactive(_update_image)
    def _on_scroll_up (self, event: tk.Event):
        step = .1
        self.world_position  -= step * np.array([0, 0, 1])
    
    @reactive(_update_image)
    def _on_press_key(self, event: tk.Event):
        step = .1
        if event.keysym == 'Escape':
            self.running = False 

        elif event.keysym in ("h", "H"):
            self._show_instructions()

        elif event.keysym in ("space",):
            self.world_position += step * np.array([0, 1, 0])

        elif event.keysym in ("z", "Z"):
            self.world_position -= step * np.array([0, 1, 0])

        elif event.keysym in ("w", "W"):
            self.world_position -= step * np.array([0, 0, 1])

        elif event.keysym in ("s", "S"):
            self.world_position += step * np.array([0, 0, 1])

        elif event.keysym in ("a", "A"):
            self.world_position += step * np.array([1, 0, 0])

        elif event.keysym in ("d", "D"):
            self.world_position -= step * np.array([1, 0, 0])

        elif event.keysym in ("e", "E"):
            rotation = rotation_z(step)
            self.world_position = (rotation @ self.world_position).astype(np.float64)
            self.world_z = (rotation @ self.world_z).astype(np.float64)
            self.world_x = (rotation @ self.world_x).astype(np.float64)
            
        elif event.keysym in ("q", "Q"):
            rotation = rotation_z(-step)
            self.world_position = (rotation @ self.world_position).astype(np.float64)
            self.world_z = (rotation @ self.world_z).astype(np.float64)
            self.world_x = (rotation @ self.world_x).astype(np.float64)

        elif event.keysym in ("Left",):
            step = 1e-1
            rotation = rotation_y(step)
            self.world_z = (rotation @ self.world_z).astype(np.float64)
            self.world_x = (rotation @ self.world_x).astype(np.float64)
            
        elif event.keysym in ("Right",):
            step = 1e-1
            rotation = rotation_y(-step)
            self.world_z = (rotation @ self.world_z).astype(np.float64)
            self.world_x = (rotation @ self.world_x).astype(np.float64)

        elif event.keysym in ("Up",):
            step = 1e-1
            rotation = rotation_x(-step)
            self.world_z = (rotation @ self.world_z).astype(np.float64)
            self.world_x = (rotation @ self.world_x).astype(np.float64)
            
        elif event.keysym in ("Down",):
            step = 1e-1
            rotation = rotation_x(step)
            self.world_z = (rotation @ self.world_z).astype(np.float64)
            self.world_x = (rotation @ self.world_x).astype(np.float64)

        elif event.keysym in ("q", "Q"):
            rotation = rotation_z(-step)
            self.world_position = (rotation @ self.world_position).astype(np.float64)
            self.world_z = (rotation @ self.world_z).astype(np.float64)
            self.world_x = (rotation @ self.world_x).astype(np.float64)

    def _release_mouse(self, event: tk.Event) -> None:
        self.mouse_x, self.mouse_y = None, None

    def _drag_mouse(self, event: tk.Event) -> Optional[tuple[int, int]]:
        """Drags the mouse.
        Takes care of updating the mouse parameters and initial conditions.
        Returns:
            None if it was the first contact.
            Tuple[int, int] with the dislocation (dx, dy) of the mouse.
        """
        if self.mouse_x is None or self.mouse_y is None:
            self.mouse_x, self.mouse_y = event.x, event.y
            return
        
        dx = event.x - self.mouse_x
        dy = event.y - self.mouse_y
        self.mouse_x, self.mouse_y = event.x, event.y
        return dx, dy
