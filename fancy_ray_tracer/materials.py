from __future__ import annotations

import numpy as np

from .constants import ATOL
from .utils import equal


class Material:  # pylint: disable=too-few-public-methods,too-many-arguments
    __slots__ = ("color", "ambient", "diffuse", "specular", "shininess")

    def __init__(self, color: np.ndarray, ambient: float,
                 diffuse: float, specular: float, shininess: float):
        self.color: np.ndarray = color
        self.ambient: float = ambient
        self.diffuse: float = diffuse
        self.specular: float = specular
        self.shininess: float = shininess

    def __eq__(self, other: Material) -> bool:
        if not isinstance(other, Material):
            raise NotImplementedError

        return equal(self.color, other.color) and abs(self.ambient - other.ambient) < ATOL \
            and abs(self.diffuse - other.diffuse) < ATOL \
            and abs(self.specular - other.specular) < ATOL \
            and abs(self.shininess - other.shininess) < ATOL


def make_material() -> Material:
    return Material(np.array([1.0, 1.0, 1.0]), 0.1, 0.9, 0.9, 200.0)
