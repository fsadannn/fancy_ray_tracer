from __future__ import annotations

import numpy as np

from .constants import EPSILON
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

        return equal(self.color, other.color) and abs(self.ambient - other.ambient) < EPSILON \
            and abs(self.diffuse - other.diffuse) < EPSILON \
            and abs(self.specular - other.specular) < EPSILON \
            and abs(self.shininess - other.shininess) < EPSILON


def make_material() -> Material:
    return Material(np.array([1.0, 1.0, 1.0]), 0.1, 0.9, 0.9, 200.0)
