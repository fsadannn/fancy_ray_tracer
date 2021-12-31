import multiprocessing
from math import sqrt, tan
from typing import Optional

import numpy as np
from joblib import Parallel, delayed

from fancy_ray_tracer.protocols import CanvasP

from .matrices import inverse
from .ray import Ray
from .world import World


class Camera:
    __slots__ = ("hsize", "vsize", "field_of_view",
                 "transform", "inv_transform",
                 "half_width", "half_height", "pixel_size")

    def __init__(self, hsize: int, vsize: int, field_of_view: float,
                 transform: Optional[np.ndarray] = None) -> None:
        self.hsize: int = hsize
        self.vsize: int = vsize
        self.field_of_view: float = field_of_view
        if transform is None:
            self.transform = np.eye(4, 4)
            self.inv_transform = self.transform
        else:
            self.transform = transform
            self.inv_transform = inverse(transform)
        half_view = tan(self.field_of_view / 2)
        aspect = self.hsize / self.vsize
        if aspect >= 1:
            self.half_width = half_view
            self.half_height = half_view / aspect
        else:
            self.half_width = half_view * aspect
            self.half_height = half_view

        self.pixel_size = (self.half_width * 2) / self.hsize

    def set_transform(self, transform: np.ndarray) -> None:
        self.transform = transform
        self.inv_transform = inverse(transform)

    def ray_for_pixel(self, px: int, py: int) -> Ray:
        xoffset: float = (px + 0.5) * self.pixel_size
        yoffset: float = (py + 0.5) * self.pixel_size
        world_x: float = self.half_width - xoffset
        world_y: float = self.half_height - yoffset

        # pixel: np.ndarray = self.inv_transform.dot(point(world_x, world_y, -1))
        pixel: np.ndarray = self.inv_transform.dot((world_x, world_y, -1, 1))
        origin = self.inv_transform[:, 3]
        direction: np.ndarray = pixel - origin
        nm: float = sqrt(direction.dot(direction))
        direction *= 1 / nm
        return Ray(origin, direction)

    def render(self, world: World, canvas: CanvasP):
        ncpu = multiprocessing.cpu_count()
        pp = Parallel(n_jobs=ncpu, verbose=10)
        c = pp(delayed(self._render_y)(world, y) for y in range(self.vsize))
        for y in range(self.vsize):
            for x in range(self.hsize):
                canvas.set_pixelf(x, y, c[y][x])

    def _render_y(self, world: World, y: int):
        cc = []
        for x in range(self.hsize):
            r = self.ray_for_pixel(x, y)
            c = world.color_at(r)
            cc.append(c)
        return cc

    def render_sequential(self, world: World, canvas: CanvasP):
        for y in range(self.vsize):
            for x in range(self.hsize):
                r = self.ray_for_pixel(x, y)
                c = world.color_at(r)
                canvas.set_pixelf(x, y, c)
