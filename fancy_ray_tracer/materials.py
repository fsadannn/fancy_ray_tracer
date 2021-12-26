from __future__ import annotations

from math import floor, pow, sqrt
from typing import Optional

import numpy as np

from .constants import EPSILON
from .matrices import identity
from .protocols import MaterialP, Pattern
from .utils import equal


class DefaultPattern(Pattern):
    __slots__ = ()

    def __init__(self) -> None:
        self.name = 'default'
        self.transform: np.ndarray = identity()
        self.inv_transform: np.ndarray = self.transform


class StripePattern(DefaultPattern):
    __slots__ = ("_c1", "_c2")

    def __init__(self, color1=np.ndarray, color2=np.ndarray) -> None:
        self.name = 'stripe'
        super().__init__()
        self._c1 = color1
        self._c2 = color2

    def color_at(self, point: np.ndarray) -> np.ndarray:
        if floor(point[0]) % 2 == 0:
            return self._c1

        return self._c2

    def __eq__(self, other: StripePattern) -> bool:
        return self.name == other.name and equal(self._c1, other._c1) and equal(self._c2, other._c2)


class LinearGradient(DefaultPattern):
    __slots__ = ("_c1", "_c2", "_gap")

    def __init__(self, color1=np.ndarray, color2=np.ndarray) -> None:
        super().__init__()
        self.name = 'gradient'
        self._c1 = color1
        self._c2 = color2
        self._gap = color2 - color1

    def color_at(self, point: np.ndarray) -> np.ndarray:
        return self._c1 + self._gap * (point[0] - floor(point[0]))

    def __eq__(self, other: StripePattern) -> bool:
        return self.name == other.name and equal(self._c1, other._c1) and equal(self._c2, other._c2)


class RingPatter(DefaultPattern):
    __slots__ = ("_c1", "_c2")

    def __init__(self, color1=np.ndarray, color2=np.ndarray) -> None:
        self.name = 'stripe'
        super().__init__()
        self._c1 = color1
        self._c2 = color2

    def color_at(self, point: np.ndarray) -> np.ndarray:
        if floor(sqrt(pow(point[0], 2) + pow(point[2], 2))) % 2 == 0:
            return self._c1

        return self._c2

    def __eq__(self, other: StripePattern) -> bool:
        return self.name == other.name and equal(self._c1, other._c1) and equal(self._c2, other._c2)


class ChessPattern(DefaultPattern):
    __slots__ = ("_c1", "_c2")

    def __init__(self, color1=np.ndarray, color2=np.ndarray) -> None:
        self.name = 'stripe'
        super().__init__()
        self._c1 = color1
        self._c2 = color2

    def color_at(self, point: np.ndarray) -> np.ndarray:
        if (floor(point[0]) + floor(point[1]) + floor(point[2])) % 2 == 0:
            return self._c1

        return self._c2

    def __eq__(self, other: StripePattern) -> bool:
        return self.name == other.name and equal(self._c1, other._c1) and equal(self._c2, other._c2)


class Material(MaterialP):  # pylint: disable=too-few-public-methods,too-many-arguments
    __slots__ = ("color", "ambient", "diffuse",
                 "specular", "shininess", "reflective", "pattern")

    def __init__(self, color: np.ndarray, ambient: float,
                 diffuse: float, specular: float, shininess: float, reflective: float = 0, pattern: Optional[Pattern] = None):
        self.color: np.ndarray = color
        self.ambient: float = ambient
        self.diffuse: float = diffuse
        self.specular: float = specular
        self.shininess: float = shininess
        self.reflective = reflective
        self.pattern: Optional[Pattern] = pattern

    def color_at(self, point: np.ndarray) -> np.ndarray:
        if self.pattern is None:
            return self.color

        point = self.pattern.inv_transform.dot(point)
        return self.pattern.color_at(point)

    def __eq__(self, other: Material) -> bool:
        if not isinstance(other, Material):
            raise NotImplementedError

        return equal(self.color, other.color) and abs(self.ambient - other.ambient) < EPSILON \
            and abs(self.diffuse - other.diffuse) < EPSILON \
            and abs(self.specular - other.specular) < EPSILON \
            and abs(self.shininess - other.shininess) < EPSILON \
            and self.pattern == other.pattern


def make_material() -> Material:
    return Material(np.array([1.0, 1.0, 1.0]), 0.1, 0.9, 0.9, 200.0)
