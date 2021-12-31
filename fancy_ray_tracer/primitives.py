from __future__ import annotations

from math import fabs, sqrt
from typing import List, Optional, Sequence

import numpy as np
from numpy.core.fromnumeric import shape

from .constants import BOX_UNITARY_MAX_BOUND, BOX_UNITARY_MIN_BOUND, EPSILON, INFINITY
from .materials import make_material
from .matrices import identity
from .protocols import WorldObject
from .ray import Intersection
from .tuples import point, vector
from .utils import rand_id

try:
    from .compiled import _intersection
except ImportError:
    _intersection = None

AXIS_X_VEC = vector(1, 0, 0)
AXIS_Y_VEC = vector(0, 1, 0)
AXIS_Z_VEC = vector(0, 0, 1)
NAXIS_X_VEC = vector(-1, 0, 0)
NAXIS_Y_VEC = vector(0, -1, 0)
NAXIS_Z_VEC = vector(0, 0, -1)
ZERO = vector(0, 0, 0)


def aabb_box_intersect_fallback(bound_min: np.ndarray, bound_max: np.ndarray,
                                origin: np.ndarray, direction: np.ndarray, epsilon: float):  # smith method
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
        self.material = make_material()
        self.transform: np.ndarray = identity()
        self.inv_transform: np.ndarray = self.transform
        self.parent: Optional[WorldObject] = None

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
    __slots__ = tuple(["_normalv"])

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

        if abs(ap[1] - maxc) < EPSILON:
            return vector(0, p[1], 0)

        return vector(0, 0, p[2])

    def intersect(self, origin: np.ndarray, direction: np.ndarray) -> Sequence[Intersection]:

        it = aabb_box_intersect(
            BOX_UNITARY_MIN_BOUND, BOX_UNITARY_MAX_BOUND, origin, direction, EPSILON)
        if it is None:
            return ()
        return [Intersection(it[0], self), Intersection(it[1], self)]


class Cylinder(Shape):
    __slots__ = ("minimum", "maximum", "closed")

    def __init__(self, minimum: float = -INFINITY, maximum: float = INFINITY,
                 closed: bool = False, shapeId: Optional[str] = None):
        super().__init__(shapeId=shapeId)
        self.minimum = minimum
        self.maximum = maximum
        self.closed = closed

    def normal_at(self, p: np.ndarray) -> np.ndarray:
        d = p[0] * p[0] + p[2] * p[2]

        if d < 1:
            if p[1] > self.maximum - EPSILON:
                return AXIS_Y_VEC
            if p[1] < self.minimum + EPSILON:
                return NAXIS_Y_VEC
        rv = p.copy()
        rv[1] = 0
        return rv

    def intersect(self, origin: np.ndarray, direction: np.ndarray) -> Sequence[Intersection]:
        a = direction[0] * direction[0] + direction[2] * direction[2]

        if a < EPSILON:
            if not self.closed:
                return ()
            # here we have a perpendicular ray to the caps
            # test if the ray hit one of the caps and if so the the second
            # cap is hit too
            xs: List[Intersection] = []
            o1 = origin[1]
            d1 = direction[1]
            d1i = 1 / d1
            t = (self.minimum - o1) * d1i
            x = origin[0] + t * direction[0]
            z = origin[2] + t * direction[2]
            if x * x + z * z <= 1.0:
                xs.append(Intersection(t, self))
            else:
                return ()

            t = (self.maximum - o1) * d1i
            x = origin[0] + t * direction[0]
            z = origin[2] + t * direction[2]
            xs.append(Intersection(t, self))

            return xs

        b = 2 * (origin[0] * direction[0] + origin[2] * direction[2])
        c = origin[0] * origin[0] + origin[2] * origin[2] - 1

        disc = b * b - 4 * a * c

        if disc < 0:
            return ()

        sqdc = sqrt(disc)
        a21 = 1 / (2 * a)
        t0 = (-b - sqdc) * a21
        t1 = (-b + sqdc) * a21

        if t0 > t1:
            t0, t1 = t1, t0

        xs: List[Intersection] = []

        o1 = origin[1]
        d1 = direction[1]
        y = o1 + t0 * d1
        if self.minimum < y < self.maximum:
            xs.append(Intersection(t0, self))
        y = o1 + t1 * d1
        if self.minimum < y < self.maximum:
            xs.append(Intersection(t1, self))

        # since cylinder is cuadric surfece can only by intrecepted at maximun of two
        # point at same time

        if not self.closed or len(xs) == 2 or abs(d1) < EPSILON:
            if len(xs) == 2 and xs[0].t > xs[1].t:
                return [xs[1], xs[0]]

            return xs

        d1i = 1 / d1
        t = (self.minimum - o1) * d1i
        x = origin[0] + t * direction[0]
        z = origin[2] + t * direction[2]
        if x * x + z * z <= 1:
            xs.append(Intersection(t, self))

        if len(xs) == 2:
            return [xs[1], xs[0]]

        t = (self.maximum - o1) * d1i
        x = origin[0] + t * direction[0]
        z = origin[2] + t * direction[2]
        if x * x + z * z <= 1:
            xs.append(Intersection(t, self))

        if len(xs) == 2 and xs[0].t > xs[1].t:
            return [xs[1], xs[0]]

        return xs


class Cone(Shape):
    __slots__ = ("minimum", "maximum", "closed", "minimum2", "maximum2")

    def __init__(self, minimum: float = -INFINITY, maximum: float = INFINITY,
                 closed: bool = False, shapeId: Optional[str] = None):
        super().__init__(shapeId=shapeId)
        self.minimum = minimum
        self.minimum2 = minimum * minimum
        self.maximum = maximum
        self.maximum2 = maximum * maximum
        self.closed = closed

    def normal_at(self, p: np.ndarray) -> np.ndarray:
        d = p[0] * p[0] + p[2] * p[2]

        if d < self.maximum2 and p[1] > self.maximum - EPSILON:
            return AXIS_Y_VEC
        if d < self.minimum2 and p[1] < self.minimum + EPSILON:
            return NAXIS_Y_VEC

        if p[1] > 0:
            return vector(p[0], -sqrt(d), p[2])

        return vector(p[0], sqrt(d), p[2])

    def intersect(self, origin: np.ndarray, direction: np.ndarray) -> Sequence[Intersection]:
        dy = direction[1]
        oy = origin[1]
        a = direction[0] * direction[0] + direction[2] * direction[2] - dy * dy
        b = 2 * (origin[0] * direction[0] + origin[2] * direction[2] - oy * dy)
        c = origin[0] * origin[0] + origin[2] * origin[2] - oy * oy

        if abs(a) < EPSILON:
            t1 = -c / (2 * b)

            if abs(b) < EPSILON:
                return ()

            if not self.closed:
                return [Intersection(t1, self)]

            dyi = 1 / dy
            t = (self.minimum - oy) * dyi
            x = origin[0] + t * direction[0]
            z = origin[2] + t * direction[2]
            if x * x + z * z <= self.minimum2:
                if t < t1:
                    return [Intersection(t, self), Intersection(t1, self)]
                return [Intersection(t1, self), Intersection(t, self)]

            t = (self.maximum - oy) * dyi
            x = origin[0] + t * direction[0]
            z = origin[2] + t * direction[2]
            if x * x + z * z <= self.maximum2:
                if t < t1:
                    return [Intersection(t, self), Intersection(t1, self)]
                return [Intersection(t1, self), Intersection(t, self)]

            return ()

        disc = b * b - 4 * a * c

        if disc < 0:
            return ()

        sqdc = sqrt(disc)
        a21 = 1 / (2 * a)
        t0 = (-b - sqdc) * a21
        t1 = (-b + sqdc) * a21

        xs: List[Intersection] = []

        o1 = origin[1]
        d1 = direction[1]
        y = o1 + t0 * d1
        if self.minimum < y < self.maximum:
            xs.append(Intersection(t0, self))
        y = o1 + t1 * d1
        if self.minimum < y < self.maximum:
            xs.append(Intersection(t1, self))

        if not self.closed:
            return xs

        dyi = 1 / dy
        t = (self.minimum - oy) * dyi
        x = origin[0] + t * direction[0]
        z = origin[2] + t * direction[2]
        if x * x + z * z <= self.minimum2:
            xs.append(Intersection(t, self))

        t = (self.maximum - oy) * dyi
        x = origin[0] + t * direction[0]
        z = origin[2] + t * direction[2]
        if x * x + z * z <= self.maximum2:
            xs.append(Intersection(t, self))

        if len(xs) < 2:
            return xs

        xs.sort()
        return xs


class Group(Shape):
    __slots__ = ("shapes")

    def __init__(self, shapes: Optional[Sequence[WorldObject]] = (), shapeId: Optional[str] = None):
        super().__init__(shapeId=shapeId)
        self.shapes: List[WorldObject] = list(shapes)

    def add_shape(self, shape: WorldObject):
        shape.parent = self
        self.shapes.append(shape)

    def normal_at(self, p: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def intersect(self, origin: np.ndarray, direction: np.ndarray) -> Sequence[Intersection]:
        xs: List[Intersection] = []

        it_count: int = 0
        for shape in self.shapes:
            iv = shape.inv_transform
            o: np.ndarray = iv.dot(origin)
            d: np.ndarray = iv.dot(direction)
            its = shape.intersect(o, d)
            it_count += len(its) > 0
            xs.extend(its)

        if len(xs) < 2 or it_count < 2:
            return xs

        xs.sort()

        return xs


class BoundingBox(Shape):
    __slots__ = ("bound_min", "bound_max", "shape")

    def __init__(self, bound_min: np.ndarray, bound_max: np.ndarray, shape: WorldObject, shapeId: Optional[str] = None):
        super().__init__(shapeId=shapeId)
        self.bound_max: np.ndarray = bound_max
        self.bound_min: np.ndarray = bound_min
        self.shape: WorldObject = shape

    def set_transform(self, transform: np.ndarray) -> None:
        super().set_transform(transform)
        self.shape.transform = self.transform
        self.shape.inv_transform = self.inv_transform

    def normal_at(self, p: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def intersect(self, origin: np.ndarray, direction: np.ndarray) -> Sequence[Intersection]:
        it = aabb_box_intersect(
            BOX_UNITARY_MIN_BOUND, BOX_UNITARY_MAX_BOUND, origin, direction, EPSILON)
        if it is None:
            return ()
        return self.shape.intersect(origin, direction)


def make_box(shape: WorldObject) -> BoundingBox:
    if isinstance(shape, Sphere):
        return BoundingBox(BOX_UNITARY_MIN_BOUND, BOX_UNITARY_MAX_BOUND, shape)

    if isinstance(shape, Cube):
        return BoundingBox(BOX_UNITARY_MIN_BOUND, BOX_UNITARY_MAX_BOUND, shape)

    if isinstance(shape, Plane):
        minb: np.ndarray = BOX_UNITARY_MIN_BOUND.copy()
        maxb: np.ndarray = BOX_UNITARY_MAX_BOUND.copy()
        minb[0] = -np.inf
        minb[1] = 0
        minb[2] = -np.inf
        maxb[0] = np.inf
        maxb[1] = 0
        maxb[2] = np.inf
        return BoundingBox(minb, maxb, shape)

    if isinstance(shape, Cylinder):
        minb: np.ndarray = BOX_UNITARY_MIN_BOUND.copy()
        maxb: np.ndarray = BOX_UNITARY_MAX_BOUND.copy()
        if shape.closed:
            minb[1] = shape.minimum
            maxb[1] = shape.maximum
        else:
            minb[1] = -np.inf
            maxb[1] = np.inf
        return BoundingBox(minb, maxb, shape)

    if isinstance(shape, Cone):
        minb: np.ndarray = BOX_UNITARY_MIN_BOUND.copy()
        maxb: np.ndarray = BOX_UNITARY_MAX_BOUND.copy()
        if shape.closed:
            xz = max(shape.minimum2, shape.maximum2)
            minb[0] = -xz
            minb[1] = shape.minimum
            minb[2] = -xz
            maxb[0] = -xz
            maxb[1] = shape.maximum
            maxb[2] = -xz
        else:
            minb[0] = -np.inf
            minb[1] = -np.inf
            minb[2] = -np.inf
            maxb[0] = np.inf
            maxb[1] = np.inf
            maxb[2] = np.inf
        return BoundingBox(minb, maxb, shape)

    if isinstance(shape, Group):
        maxx = -np.inf
        maxy = -np.inf
        maxz = -np.inf
        minx = np.inf
        miny = np.inf
        minz = np.inf
        for sh in shape.shapes:
            b = make_box(sh)
            pmin = sh.transform.dot(b.bound_min)
            pmax = sh.transform.dot(b.bound_max)
            maxx = max(pmax[0], maxx)
            maxy = max(pmax[1], maxy)
            maxz = max(pmax[2], maxz)
            minx = min(pmin[0], minx)
            miny = min(pmin[1], miny)
            minz = min(pmin[2], minz)
        pmin = point(minx, miny, minz)
        pmax = point(maxx, maxy, maxz)
        return BoundingBox(pmin, pmax, shape)
