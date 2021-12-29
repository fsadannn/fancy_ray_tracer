from __future__ import annotations

from math import fabs
from math import inf as INFINITY
from math import sqrt
from typing import Optional, Sequence, Tuple

import numpy as np

from .constants import EPSILON
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


def check_axis_fallback(origin: float, direction: float, epsilon: float, axis_min: float = -1.0, axis_max: float = 1.0) -> Tuple[float, float]:
    if fabs(direction) >= epsilon:
        tmin = (axis_min - origin) / direction
        tmax = (axis_max - origin) / direction
    else:
        tmin = (axis_min - origin) * INFINITY
        tmax = (axis_max - origin) * INFINITY

    if tmin > tmax:
        temp = tmin
        tmin = tmax
        tmax = temp

    return (tmin, tmax)


check_axis = _intersection.check_axis if _intersection is not None else check_axis_fallback


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
        xtmin, xtmax = check_axis(origin[0], direction[0], EPSILON)
        ytmin, ytmax = check_axis(origin[1], direction[1], EPSILON)
        ztmin, ztmax = check_axis(origin[2], direction[2], EPSILON)
        tmin = max(xtmin, ytmin, ztmin)
        tmax = min(xtmax, ytmax, ztmax)
        if tmin > tmax:
            return ()
        return [Intersection(tmin, self), Intersection(tmax, self)]
