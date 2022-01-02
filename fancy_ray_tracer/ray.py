from __future__ import annotations

from bisect import bisect_left
from functools import total_ordering
from math import sqrt
from typing import List, Optional, Sequence

import numpy as np

from .constants import EPSILON
from .protocols import WorldObject
# from .tuples import normalize, point
from .utils import equal


@total_ordering
class Intersection:
    __slots__ = ("t", "object", "u", "v")

    def __init__(self, t: float, obj: WorldObject, u: Optional[float] = None, v: Optional[float] = None):
        self.t: float = t
        self.object: WorldObject = obj
        self.u = u
        self.v = v

    def __eq__(self, other: Intersection) -> bool:
        return abs(self.t - other.t) < EPSILON and self.object == other.object

    def __lt__(self, other: Intersection) -> bool:
        return self.t < other.t

    def __repr__(self) -> str:
        return f"<Intersection t={self.t} obj=<{str(self.object.__class__.__name__)} {self.object.id}>>"


class Ray:
    __slots__ = ("origin", "direction")

    def __init__(self, origin: np.ndarray, direction: np.ndarray):
        self.origin: np.ndarray = origin
        self.direction: np.ndarray = direction

    def position(self, t: float) -> np.ndarray:
        return self.origin + self.direction * t

    def __eq__(self, other: Ray) -> bool:
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
        return s.intersect(origin, direction)


class Computations:
    __slots__ = ("t", "object", "point", "eyev",
                 "normalv", "inside", "over_point", "reflectv",
                 "n1", "n2", "under_point")

    def __init__(self, intersection: Intersection, ray: Ray, xs: Sequence[Intersection] = ()) -> None:
        self.t: float = intersection.t
        self.object: WorldObject = intersection.object
        self.point: np.ndarray = ray.origin + ray.direction * self.t
        self.eyev = -ray.direction
        self.normalv = normal_at(self.object, self.point, intersection)
        if self.normalv.dot(self.eyev) < 0:
            self.inside: bool = True
            self.normalv = -self.normalv
        else:
            self.inside = False
        self.over_point: np.ndarray = self.point + self.normalv * EPSILON
        self.under_point: np.ndarray = self.point - self.normalv * EPSILON
        self.reflectv = ray.direction - \
            (2 * ray.direction.dot(self.normalv)) * self.normalv

        n1 = 1.0
        n2 = 1.0
        containers: List[WorldObject] = []
        i: Intersection
        isHit: bool = False
        obj: WorldObject = None
        for i in xs:
            isHit = i == intersection

            if isHit:
                if len(containers) == 0:
                    n1 = 1.0
                else:
                    n1 = containers[-1].material.refractive_index

            obj = i.object
            if obj in containers:
                containers.remove(obj)
            else:
                containers.append(obj)

            if isHit:
                if len(containers) == 0:
                    n2 = 1.0
                else:
                    n2 = containers[-1].material.refractive_index
                break

        self.n1 = n1
        self.n2 = n2


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
    if intersections is None or len(intersections) == 0:
        return None

    temp = Intersection(0, None)

    index = bisect_left(intersections, temp)

    if index >= len(intersections):
        return None

    return intersections[index]


def normal_at(obj: WorldObject, p: np.ndarray, it: Intersection) -> np.ndarray:
    if obj.parent is None:
        tinv = obj.inv_transform
        object_point: np.ndarray = tinv.dot(p)
        object_normal: np.ndarray = obj.normal_at(object_point, it)
        world_normal: np.ndarray = tinv.T.dot(object_normal)
        world_normal[3] = 0.0
        # normalize inplace
        # world_normal = normalize(world_normal)
        nm: float = sqrt(world_normal.dot(world_normal))
        world_normal *= (1.0 / nm)
        return world_normal

    object_point = world_to_object(obj, p)
    object_normal: np.ndarray = obj.normal_at(object_point)
    world_normal = normal_to_world(obj, object_normal)
    return world_normal


def world_to_object(shape: WorldObject, p: np.ndarray) -> np.ndarray:
    if shape.parent is not None:
        p = world_to_object(shape.parent, p)

    return shape.inv_transform.dot(p)


def normal_to_world(shape: WorldObject, normal: np.ndarray) -> np.ndarray:
    normal = shape.inv_transform.T.dot(normal)
    normal[3] = 0.0
    nm: float = sqrt(normal.dot(normal))
    normal *= (1.0 / nm)
    if shape.parent is not None:
        normal = normal_to_world(shape.parent, normal)

    return normal
