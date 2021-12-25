from __future__ import annotations

from bisect import bisect_left
from functools import total_ordering
from math import pow, sqrt
from typing import Optional, Sequence

import numpy as np

from .constants import ATOL
from .protocols import WorldObject
# from .tuples import normalize, point
from .utils import equal


@total_ordering
class Intersection:
    __slots__ = ("t", "object")

    def __init__(self, t: float, object: WorldObject):
        self.t: float = t
        self.object: WorldObject = object

    def __eq__(self, other: Intersection) -> bool:
        if not isinstance(other, Intersection):
            raise NotImplementedError

        return abs(self.t - other.t) < ATOL and self.object == other.object

    def __lt__(self, other: Intersection):
        if not isinstance(other, Intersection):
            raise NotImplementedError

        return self.t < other.t


class Ray:
    __slots__ = ("origin", "direction")

    def __init__(self, origin: np.ndarray, direction: np.ndarray):
        self.origin: np.ndarray = origin
        self.direction: np.ndarray = direction

    def position(self, t: float) -> np.ndarray:
        return self.origin + self.direction * t

    def __eq__(self, other: Ray) -> bool:
        if not isinstance(other, Ray):
            raise NotImplementedError

        return equal(self.direction, other.direction) and equal(self.origin, other.origin)

    def transform(self, matrix: np.ndarray) -> Ray:
        orig: np.ndarray = matrix.dot(self.origin)
        direct: np.ndarray = matrix.dot(self.direction)
        return Ray(orig, direct)

    def intersect(self, s: WorldObject) -> Sequence[Intersection]:
        # tranform the ray, equivalent to self.transform
        invt = s.inv_transform
        origin: np.ndarray = invt.dot(self.origin)
        direction: np.ndarray = invt.dot(self.direction)

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


class Computations:
    __slots__ = ("t", "object", "point")

    def __init__(self, intersection: Intersection, ray: Ray) -> None:
        self.t: float = intersection.t
        self.object: WorldObject = intersection.object
        self.point: np.ndarray = ray.position(intersection.t)


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


def hit_sorted(intersections: Sequence[Intersection]) -> Optional[Intersection]:
    if intersections is None or len(intersections == 0):
        return None

    temp = Intersection(0, None)

    index = bisect_left(intersections, temp)

    return intersections[index]


def normal_at(object: WorldObject, p: np.ndarray) -> np.ndarray:
    tinv = object.inv_transform
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
