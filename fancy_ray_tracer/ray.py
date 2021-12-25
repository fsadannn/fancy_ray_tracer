from __future__ import annotations

from math import pow, sqrt
from typing import Optional, Sequence

import numpy as np

from .constants import ATOL
from .materials import Material
from .protocols import WorldObject
from .tuples import normalize, point
from .utils import equal


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


class Light:
    __slots__ = ("_intensity", "_position")

    def __init__(self, position: np.ndarray, intensity: np.ndarray):
        self._intensity: np.ndarray = intensity
        self._position: np.ndarray = position

    @property
    def intensity(self) -> np.ndarray:
        return self._intensity

    @property
    def position(self) -> np.ndarray:
        return self._position

    def __eq__(self, other: Light) -> bool:
        if not isinstance(other, Light):
            raise NotImplementedError

        return equal(self._position, other._position) and equal(self._intensity, other._intensity)


def intersect(s: WorldObject, r: Ray) -> Sequence[Intersection]:
    ray2 = transform(r, s.inv_transform)

    sphere_to_ray = ray2.origin - point(0.0, 0.0, 0.0)
    a = ray2.direction.dot(ray2.direction)
    b = 2 * ray2.direction.dot(sphere_to_ray)
    c = sphere_to_ray.dot(sphere_to_ray) - 1
    dc = b**2 - 4 * a * c

    if dc < 0:
        return ()

    dcsq = sqrt(dc)

    r1 = (-b - dcsq) / (2 * a)
    r2 = (-b + dcsq) / (2 * a)

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


def transform(ray: Ray, matrix: np.ndarray) -> Ray:
    orig: np.ndarray = matrix.dot(ray._origin)
    direct: np.ndarray = matrix.dot(ray._direction)
    return Ray(orig, direct)


def normal_at(object: WorldObject, p: np.ndarray) -> np.ndarray:
    tinv = object.inv_transform
    object_point = tinv.dot(p)
    object_normal = object_point - point(0.0, 0.0, 0.0)
    world_normal = tinv.T.dot(object_normal)
    world_normal[3] = 0
    world_normal = normalize(world_normal)
    return world_normal


def reflect(v: np.ndarray, n: np.ndarray) -> np.ndarray:
    return v - (2 * v.dot(n)) * n


def lighting(material: Material, light: Light, point: np.ndarray,
             eyev: np.ndarray, normalv: np.ndarray):
    # combine the surface color with the light's color/intensity
    effective_color = material.color * light.intensity
    # find the direction to the light source
    lightv = normalize(light.position - point)
    # compute the ambient contribution
    ambient = effective_color * material.ambient
    # light_dot_normal represents the cosine of the angle between the
    # light vector and the normal vector. A negative number means the
    # light is on the other side of the surface.
    light_dot_normal: float = lightv.dot(normalv)
    if light_dot_normal < 0:
        diffuse = 0
        specular = 0
    else:
        # compute the diffuse contribution
        diffuse = effective_color * material.diffuse * light_dot_normal
        # reflect_dot_eye represents the cosine of the angle between the
        # reflection vector and the eye vector. A negative number means the
        # light reflects away from the eye.
        reflectv = reflect(-lightv, normalv)
        reflect_dot_eye: float = reflectv.dot(eyev)

        if reflect_dot_eye <= 0:
            specular = 0
        else:
            # compute the specular contribution
            factor = pow(reflect_dot_eye, material.shininess)
            specular = light.intensity * material.specular * factor

    # Add the three contributions together to get the final shading
    return ambient + diffuse + specular
