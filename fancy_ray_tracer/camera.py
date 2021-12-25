from math import tan
from typing import Optional

import numpy as np


class Camera:
    __slots__ = ("hsize", "vsize", "field_of_view",
                 "transform", "half_width", "half_height", "pixel_size")

    def __init__(self, hsize: int, vsize: int, field_of_view: float, transform: Optional[np.ndarray] = None) -> None:
        self.hsize: int = hsize
        self.vsize: int = vsize
        self.field_of_view: float = field_of_view
        self.transform = transform if transform else np.eye(4, 4)
        half_view = tan(self.field_of_view / 2)
        aspect = self.hsize / self.vsize
        if aspect >= 1:
            self.half_width = half_view
            self.half_height = half_view / aspect
        else:
            self.half_width = half_view * aspect
            self.half_height = half_view

        self.pixel_size = (self.half_width * 2) / self.hsize
