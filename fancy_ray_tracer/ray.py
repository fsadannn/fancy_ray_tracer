from __future__ import annotations

from math import pow, sqrt
from typing import Optional, Sequence

import numpy as np

from .constants import ATOL
from .protocols import WorldObject
# from .tuples import normalize, point
from .utils import equal


class Intersection:
    __slots__ = ("_t", "_object")

    def __init__(self, t: float, object: WorldObject):
        self._t: float = t
        self._object: WorldObject = object

    @property
    def object(self) -> WorldObject:
        return self._object

    @property
    def t(self) -> float:
        return self._t

    def __eq__(self, other: Intersection) -> bool:
        if not isinstance(other, Intersection):
            raise NotImplementedError

        return abs(self._t - other._t) < ATOL and self._object == other._object


class Ray:
    __slots__ = ("_origin", "_direction")

    def __init__(self, origin: np.ndarray, direction: np.ndarray):
        self._origin: np.ndarray = origin
        self._direction: np.ndarray = direction

    @property
    def origin(self) -> np.ndarray:
        return self._origin

    @property
    def direction(self) -> np.ndarray:
        return self._direction

    def position(self, t: float) -> np.ndarray:
        return self._origin + self._direction * t

    def __eq__(self, other: Ray) -> bool:
        if not isinstance(other, Ray):
            raise NotImplementedError

        return equal(self._direction, other._direction) and equal(self._origin, other._origin)

    def transform(self, matrix: np.ndarray) -> Ray:
        orig: np.ndarray = matrix.dot(self._origin)
        direct: np.ndarray = matrix.dot(self._direction)
        return Ray(orig, direct)

    def intersect(self, s: WorldObject) -> Sequence[Intersection]:
        # tranform the ray, equivalent to self.transform
        invt = s._inv_transform
        origin: np.ndarray = invt.dot(self._origin)
        direction: np.ndarray = invt.dot(self._direction)

        # sphere_to_ray = origin - point(0.0, 0.0, 0.0)
        sphere_to_ray = origin.copy()
        sphere_to_ray[3] = 0.0

        a: float = direction.dot(direction)
        b: float = 2.0 * direction.dot(sphere_to_ray)
        c: float = sphere_to_ray.dot(sphere_to_ray) - 1
        dc = pow(b, 2.0) - 4.0 * a * c

        if dc < 0:
            return ()

        dcsq = sqrt(dc)
        a12 = 1.0 / (2.0 * a)

        r1 = (-b - dcsq) * a12
        r2 = (-b + dcsq) * a12

        return Intersection(r1, s), Intersection(r2, s)


def hit(intersections: Sequence[Intersection]) -> Optional[Intersection]:
    it: Optional[Intersection] = None
    i: Intersection

    for i in intersections:
        if i.t < 0:
            continue
        if it is None:
            it = i
        elif i.t < it.t:
            it = i

    return it


def normal_at(object: WorldObject, p: np.ndarray) -> np.ndarray:
    tinv = object._inv_transform
    object_point: np.ndarray = tinv.dot(p)
    # object_normal = object_point - point(0.0, 0.0, 0.0)
    object_normal: np.ndarray = object_point.copy()
    object_normal[3] = 0.0
    world_normal: np.ndarray = tinv.T.dot(object_normal)
    world_normal[3] = 0.0
    # normalize inplace
    # world_normal = normalize(world_normal)
    nm: float = sqrt(world_normal.dot(world_normal))
    world_normal *= (1.0 / nm)
    return world_normal
