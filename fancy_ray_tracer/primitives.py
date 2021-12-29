from __future__ import annotations

from math import fabs
from math import inf as INFINITY
from math import sqrt
from typing import Optional, Sequence, Tuple

import numpy as np

from .constants import BOX_UNITARY_MAX_BOUND, BOX_UNITARY_MIN_BOUND, EPSILON
from .materials import make_material
from .matrices import identity
from .protocols import WorldObject
from .ray import Intersection
from .tuples import vector
from .utils import rand_id

try:
    from .compiled import _intersection
except ImportError:
    _intersection = None


def aabb_box_intersect_fallback(bound_min: np.ndarray, bound_max: np.ndarray, origin: np.ndarray, direction: np.ndarray, epsilon: float):  # smith method
    temp = direction[0]
    if fabs(temp) >= epsilon:
        temp = 1 / temp
    else:
        temp = INFINITY

    if temp >= 0:
        tmin = (bound_min[0] - origin[0]) * temp
        tmax = (bound_max[0] - origin[0]) * temp
    else:
        tmin = (bound_max[0] - origin[0]) * temp
        tmax = (bound_min[0] - origin[0]) * temp

    temp = direction[1]
    if fabs(temp) >= epsilon:
        temp = 1 / temp
    else:
        temp = INFINITY

    if temp >= 0:
        tminy = (bound_min[1] - origin[1]) * temp
        tmaxy = (bound_max[1] - origin[1]) * temp
    else:
        tminy = (bound_max[1] - origin[1]) * temp
        tmaxy = (bound_min[1] - origin[1]) * temp

    if tmin > tmaxy or tminy > tmax:
        return None
    if tminy > tmin:
        tmin = tminy
    if tmaxy < tmax:
        tmax = tmaxy

    temp = direction[2]
    if fabs(temp) >= epsilon:
        temp = 1 / temp
    else:
        temp = INFINITY

    if temp >= 0:
        tminz = (bound_min[2] - origin[2]) * temp
        tmaxz = (bound_max[2] - origin[2]) * temp
    else:
        tminz = (bound_max[2] - origin[2]) * temp
        tmaxz = (bound_min[2] - origin[2]) * temp

    if tmin > tmaxz or tminz > tmax:
        return None
    if tminz > tmin:
        tmin = tminz
    if tmaxz < tmax:
        tmax = tmaxz

    return (tmin, tmax)


aabb_box_intersect = _intersection.aabb_box_intersect if _intersection is not None else aabb_box_intersect_fallback


class Shape(WorldObject):
    __slots__ = ("id", 'transform', 'material', 'inv_transform')

    def __init__(self, shapeId: Optional[str] = None):
        self.id: str = shapeId if shapeId else rand_id()
        self.transform: np.ndarray = identity()
        self.material = make_material()
        self.inv_transform: np.ndarray = self.transform

    def color_at(self, point: np.ndarray) -> np.ndarray:
        point = self.inv_transform.dot(point)
        return self.material.color_at(point)

    def __eq__(self, other: WorldObject) -> bool:
        return self.id == other.id

    def normal_at(self, p: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def intersect(self, origin: np.ndarray, direction: np.ndarray) -> Sequence[Intersection]:
        raise NotImplementedError


class Sphere(Shape):
    __slots__ = ()

    def normal_at(self, p: np.ndarray) -> np.ndarray:
        p[3] = 0.0
        return p

    def intersect(self, origin: np.ndarray, direction: np.ndarray) -> Sequence[Intersection]:
        origin[3] = 0.0

        a: float = direction.dot(direction)
        b: float = 2.0 * direction.dot(origin)
        c: float = origin.dot(origin) - 1
        dc = b * b - 4.0 * a * c

        if dc < 0:
            return ()

        dcsq = sqrt(dc)
        a12 = 1.0 / (2.0 * a)

        r1 = (-b - dcsq) * a12
        r2 = (-b + dcsq) * a12

        return Intersection(r1, self), Intersection(r2, self)


class Plane(Shape):
    __slots__ = ("_normalv")

    def __init__(self, shapeId: Optional[str] = None):
        super().__init__(shapeId=shapeId)
        self._normalv: np.ndarray = vector(0, 1, 0)

    def normal_at(self, p: np.ndarray) -> np.ndarray:
        return self._normalv

    def intersect(self, origin: np.ndarray, direction: np.ndarray) -> Sequence[Intersection]:
        if abs(direction[1]) < EPSILON:
            return ()

        t = -origin[1] / direction[1]
        return [Intersection(t, self)]


def glass_sphere() -> Sphere:
    s = Sphere()
    s.material.transparency = 1.0
    s.material.refractive_index = 1.5
    return s


class Cube(Shape):
    __slots__ = ()

    def normal_at(self, p: np.ndarray) -> np.ndarray:
        ap = np.abs(p[:3])
        maxc: float = np.max(ap)
        if abs(ap[0] - maxc) < EPSILON:
            return vector(p[0], 0, 0)
        elif abs(ap[1] - maxc) < EPSILON:
            return vector(0, p[1], 0)

        return vector(0, 0, p[2])

    def intersect(self, origin: np.ndarray, direction: np.ndarray) -> Sequence[Intersection]:

        it = aabb_box_intersect(
            BOX_UNITARY_MIN_BOUND, BOX_UNITARY_MAX_BOUND, origin, direction, EPSILON)
        if it is None:
            return ()
        return [Intersection(it[0], self), Intersection(it[1], self)]
