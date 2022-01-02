from __future__ import annotations

from math import fabs, sqrt
from typing import Dict, List, Optional, Sequence

import numpy as np

from .constants import (
    BOX_UNITARY_MAX_BOUND,
    BOX_UNITARY_MIN_BOUND,
    EPSILON,
    IDENTITY,
    INFINITY,
    CSGOperation,
)
from .materials import Material, make_material
from .protocols import TriangleFaces, WorldObject
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
        self.transform: np.ndarray = IDENTITY
        self.inv_transform: np.ndarray = self.transform
        self.parent: Optional[WorldObject] = None
        self.has_shadow = True

    def color_at(self, point: np.ndarray) -> np.ndarray:
        point = self.inv_transform.dot(point)
        return self.material.color_at(point)

    def __eq__(self, other: WorldObject) -> bool:
        return self.id == other.id

    def normal_at(self, p: np.ndarray, it: Intersection = None) -> np.ndarray:
        raise NotImplementedError

    def intersect(self, origin: np.ndarray, direction: np.ndarray) -> Sequence[Intersection]:
        raise NotImplementedError

    def __contains__(self, x: WorldObject) -> bool:
        if x.id == self.id:
            return True

        return False


class Sphere(Shape):
    __slots__ = ()

    def normal_at(self, p: np.ndarray, it: Optional[Intersection] = None) -> np.ndarray:
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

    def normal_at(self, p: np.ndarray, it: Optional[Intersection] = None) -> np.ndarray:
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

    def normal_at(self, p: np.ndarray, it: Optional[Intersection] = None) -> np.ndarray:
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

    def normal_at(self, p: np.ndarray, it: Optional[Intersection] = None) -> np.ndarray:
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

    def normal_at(self, p: np.ndarray, it: Optional[Intersection] = None) -> np.ndarray:
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
        if len(self.shapes) != 0:
            for i in self.shapes:
                i.parent = self

    def add_shape(self, shape: WorldObject):
        shape.parent = self
        self.shapes.append(shape)

    def normal_at(self, p: np.ndarray, it: Optional[Intersection] = None) -> np.ndarray:
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

    def __contains__(self, x: WorldObject) -> bool:

        # traverse the hierarchy up to reach the current object or reach root object
        parent: WorldObject = x.parent
        while parent is not None:
            if parent.id == self.id:
                return True
            parent = parent.parent

        if x.id == self.id:
            return True

        return False


class BoundingBox(Shape):
    __slots__ = ("bound_min", "bound_max", "shape")

    def __init__(self, bound_min: np.ndarray, bound_max: np.ndarray, shape: WorldObject, shapeId: Optional[str] = None):
        super().__init__(shapeId=shapeId)
        self.bound_max: np.ndarray = bound_max
        self.bound_min: np.ndarray = bound_min
        self.shape: WorldObject = shape
        self.inv_transform = self.shape.inv_transform
        self.transform = self.shape.transform

    def set_transform(self, transform: np.ndarray) -> None:
        self.shape.set_transform(transform)
        self.inv_transform = self.shape.inv_transform
        self.transform = self.shape.transform

    def normal_at(self, p: np.ndarray, it: Optional[Intersection] = None) -> np.ndarray:
        raise NotImplementedError

    def intersect(self, origin: np.ndarray, direction: np.ndarray) -> Sequence[Intersection]:
        it = aabb_box_intersect(
            self.bound_min, self.bound_max, origin, direction, EPSILON)
        if it is None:
            return ()
        return self.shape.intersect(origin, direction)

    def __contains__(self, x: WorldObject) -> bool:
        # traverse the hierarchy up to reach the current object of reach root object

        if x in self.shape or x.id == self.id:
            return True

        return False


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
            xz = max(abs(shape.minimum), abs(shape.maximum))
            minb[0] = -xz
            minb[1] = shape.minimum
            minb[2] = -xz
            maxb[0] = xz
            maxb[1] = shape.maximum
            maxb[2] = xz
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
            # print(sh)
            # print(b.bound_min, pmin)
            # print(b.bound_max, pmax)

            maxx = max(pmax[0], maxx)
            maxy = max(pmax[1], maxy)
            maxz = max(pmax[2], maxz)
            minx = min(pmin[0], minx)
            miny = min(pmin[1], miny)
            minz = min(pmin[2], minz)
        pmin = point(minx, miny, minz)
        pmax = point(maxx, maxy, maxz)
        return BoundingBox(pmin, pmax, shape)

    if isinstance(shape, Triangle):
        maxx = max(shape.p1[0], shape.p2[0], shape.p3[0])
        maxy = max(shape.p1[1], shape.p2[1], shape.p3[1])
        maxz = max(shape.p1[2], shape.p2[2], shape.p3[2])
        minx = min(shape.p1[0], shape.p2[0], shape.p3[0])
        miny = min(shape.p1[1], shape.p2[1], shape.p3[1])
        minz = min(shape.p1[2], shape.p2[2], shape.p3[2])
        pmin = point(minx, miny, minz)
        pmax = point(maxx, maxy, maxz)
        return BoundingBox(pmin, pmax, shape)

    if isinstance(shape, TriangleMesh):
        vx = np.asfarray(shape.vertices)
        maxx = np.max(vx[:, 0])
        maxy = np.max(vx[:, 1])
        maxz = np.max(vx[:, 2])
        minx = np.min(vx[:, 0])
        miny = np.min(vx[:, 1])
        minz = np.min(vx[:, 2])
        pmin = point(minx, miny, minz)
        pmax = point(maxx, maxy, maxz)
        return BoundingBox(pmin, pmax, shape)

    if isinstance(shape, BoundingBox):
        return shape


class Triangle(Shape):
    __slots__ = ("p1", "p2", "p3", "e1", "e2", "normal")

    def __init__(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, shapeId: Optional[str] = None):
        super().__init__(shapeId=shapeId)
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.e1 = p2 - p1
        self.e2 = p3 - p1
        normal: np.ndarray = np.cross(self.e2[:3], self.e1[:3])
        nm = sqrt(normal.dot(normal))
        self.normal = np.append((1 / nm) * normal, 0)

    def normal_at(self, p: np.ndarray, it: Optional[Intersection] = None) -> np.ndarray:
        return self.normal

    def intersect(self, origin: np.ndarray, direction: np.ndarray) -> Sequence[Intersection]:
        direction = direction[:3]
        dir_cross_e2: np.ndarray = np.cross(direction, self.e2[:3])
        det: float = self.e1[:3].dot(dir_cross_e2)

        if abs(det) < EPSILON:
            return ()

        f: float = 1.0 / det
        p1_to_origin: np.ndarray = origin - self.p1
        p1_to_origin = p1_to_origin[:3]
        u: float = f * p1_to_origin.dot(dir_cross_e2)

        if u < 0 or u > 1:
            return ()

        origin_cross_e1: np.ndarray = np.cross(p1_to_origin, self.e1[:3])
        v: float = f * direction.dot(origin_cross_e1)
        if v < 0 or (u + v) > 1:
            return ()

        t: float = f * self.e2[:3].dot(origin_cross_e1)

        return [Intersection(t, self)]


# TODO: Implement textures
class TriangleMesh(Shape):
    __slots__ = ("vertices", "faces_groups", "normals",
                 "normals_groups", "textures", "texture_groups",
                 "e1", "e2")

    def __init__(self, vertices: List[np.ndarray], faces_group: TriangleFaces,
                 normals: List[np.ndarray], normals_group: TriangleFaces = None,
                 textures: Optional[List[np.ndarray]] = None, texture_group: Optional[TriangleFaces] = None,
                 shapeId: Optional[str] = None):
        super().__init__(shapeId=shapeId)
        self.vertices: List[np.ndarray] = vertices
        self.faces_groups: TriangleFaces = faces_group
        self.normals: List[np.ndarray] = normals
        self.normals_groups: TriangleFaces = normals_group
        self.textures: List[np.ndarray] = textures
        self.texture_groups: TriangleFaces = texture_group
        e1a = []
        e2a = []
        for face in self.faces_groups:
            p1: np.ndarray = self.vertices[face[0]]
            p2: np.ndarray = self.vertices[face[1]]
            p3: np.ndarray = self.vertices[face[2]]
            e1: np.ndarray = p2 - p1
            e1 = e1[:3]
            e1a.append(e1)
            e2: np.ndarray = p3 - p1
            e2 = e2[:3]
            e2a.append(e2)
        self.e1: np.ndarray = np.array(e1a, copy=False, dtype=np.float64)
        self.e2: np.ndarray = np.array(e2a, copy=False, dtype=np.float64)

    def normal_at(self, p: np.ndarray, it: Optional[Intersection]) -> np.ndarray:
        raise NotImplementedError

    def intersect(self, origin: np.ndarray, direction: np.ndarray) -> Sequence[Intersection]:
        xs: Sequence[Intersection] = []

        direction = direction[:3]
        e1a = self.e1
        e2a = self.e2
        dir_cross_e2a: np.ndarray = np.cross(direction, e2a, axisb=1)
        # deta: float = np.sum(e1a * dir_cross_e2a, axis=1)
        # TODO: move to cython
        for n, face in enumerate(self.faces_groups):
            p1: np.ndarray = self.vertices[face[0]]
            e1: np.ndarray = e1a[n]
            e2: np.ndarray = e2a[n]

            direction = direction[:3]
            dir_cross_e2: np.ndarray = dir_cross_e2a[n]
            # det: float = deta[n]
            det: float = e1.dot(dir_cross_e2)

            if abs(det) < EPSILON:
                continue

            f: float = 1.0 / det
            p1_to_origin: np.ndarray = origin - p1
            p1_to_origin = p1_to_origin[:3]
            u: float = f * p1_to_origin.dot(dir_cross_e2)

            if u < 0 or u > 1:
                continue

            origin_cross_e1: np.ndarray = np.cross(p1_to_origin, e1)
            v: float = f * direction.dot(origin_cross_e1)
            if v < 0 or (u + v) > 1:
                continue

            t: float = f * e2.dot(origin_cross_e1)
            nn = self.normals_groups[n]
            it = Intersection(t, SmoothTriangle(
                self.normals[nn[0]], self.normals[nn[1]], self.normals[nn[2]], self.material,), u, v)

            xs.append(it)

        if len(xs) < 2:
            return xs

        xs.sort()
        return xs


class SmoothTriangle(Shape):
    """This class is only for intersections with TriangleMesh"""
    __slots__ = ("n1", "n2", "n3")

    def __init__(self, n1: np.ndarray, n2: np.ndarray, n3: np.ndarray,
                 material: Material, shapeId: Optional[str] = None):
        # super().__init__(shapeId=shapeId)
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.transform = IDENTITY
        self.inv_transform = IDENTITY
        self.parent = None
        self.material = material
        self.id = shapeId

    def normal_at(self, p: np.ndarray, it: Intersection) -> np.ndarray:
        c = 1 - it.u - it.v
        return it.u * self.n1 + it.v * self.n2 + c * self.n3

    def intersect(self, origin: np.ndarray, direction: np.ndarray) -> Sequence[Intersection]:
        raise NotImplementedError


class CSG(Shape):
    __slot__ = ("left", "right", "op")

    def __init__(self, op: CSGOperation, left: WorldObject, right: WorldObject, shapeId: Optional[str] = None):
        left.parent = self
        right.parent = self
        super().__init__(shapeId=shapeId)
        self.left: WorldObject = left
        self.right: WorldObject = right
        self.op: CSGOperation = op

    def __contains__(self, x: WorldObject) -> bool:
        return x.id == self.id or x in self.left or x in self.right

    def normal_at(self, p: np.ndarray, it: Intersection = None) -> np.ndarray:
        raise NotImplementedError

    def intersect(self, origin: np.ndarray, direction: np.ndarray) -> Sequence[Intersection]:
        xs: List[Intersection] = []
        xs.extend(self.left.intersect(self.left.inv_transform.dot(
            origin), self.left.inv_transform.dot(direction)))
        xs.extend(self.right.intersect(self.right.inv_transform.dot(
            origin), self.right.inv_transform.dot(direction)))
        if len(xs) >= 2:
            xs.sort()

        print(xs)

        inl = False
        inr = False
        res = []
        for it in xs:
            lhit = it.object in self.left

            if interception_allowed(self.op, lhit, inl, inr):
                res.append(it)

            if lhit:
                inl = not inl
            else:
                inr = not inr

        return res


def interception_allowed(op: CSGOperation, lhit: bool, inl: bool, inr: bool) -> bool:
    if op is CSGOperation.union:
        return (lhit and not inr) or (not lhit and not inl)

    if op is CSGOperation.interception:
        return (lhit and inr) or (not lhit and inl)

    if op is CSGOperation.difference:
        return (lhit and not inr) or (not lhit and inl)

    return False


def make_csg(op: CSGOperation, shape1: WorldObject, shape2: WorldObject) -> CSG:
    return CSG(op, shape1, shape2)
