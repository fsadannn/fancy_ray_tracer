from __future__ import annotations

from math import pow, sqrt
from typing import Optional, Sequence

import numpy as np

from .matrices import identity, inverse
from .tuples import ATOL, equal, normalize, point
from .utils import rand_id


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


class Sphere:
    __slots__ = ("_id", '_transform', 'material')

    def __init__(self, sphereId: Optional[str] = None):
        self._id: str = sphereId if sphereId else rand_id()
        self._transform: np.ndarray = identity()
        self.material = make_material()

    @property
    def transform(self) -> np.ndarray:
        return self._transform

    @property
    def id(self) -> str:
        return self._id

    def set_transform(self, transform: np.ndarray):
        self._transform = transform

    def __eq__(self, other: Sphere) -> bool:
        if not isinstance(other, Sphere):
            raise NotImplementedError

        return self.id == other.id


class Intersection:
    __slots__ = ("_t", "_object")

    def __init__(self, t: float, sphere: Sphere):
        self._t: float = t
        self._object: Sphere = sphere

    @property
    def object(self) -> Sphere:
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


class Material:
    __slots__ = ("color", "ambient", "diffuse", "specular", "shininess")

    def __init__(self, color: np.ndarray, ambient: float, diffuse: float, specular: float, shininess: float):
        self.color: np.ndarray = color
        self.ambient: float = ambient
        self.diffuse: float = diffuse
        self.specular: float = specular
        self.shininess: float = shininess

    def __eq__(self, other: Material) -> bool:
        if not isinstance(other, Material):
            raise NotImplementedError

        return equal(self.color, other.color) and abs(self.ambient - other.ambient) < ATOL and abs(self.diffuse - other.diffuse) < ATOL and abs(self.specular - other.specular) < ATOL and abs(self.shininess - other.shininess) < ATOL


def make_sphere() -> Sphere:
    return Sphere()


def make_material() -> Material:
    return Material(np.array([1, 1, 1]), 0.1, 0.9, 0.9, 200.0)


def intersect(s: Sphere, r: Ray) -> Sequence[Intersection]:
    ray2 = transform(r, inverse(s.transform))

    sphere_to_ray = ray2.origin - point(0, 0, 0)
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


def normal_at(sphere: Sphere, p: np.ndarray) -> np.ndarray:
    tinv = inverse(sphere.transform)
    object_point = tinv.dot(p)
    object_normal = object_point - point(0, 0, 0)
    world_normal = tinv.T.dot(object_normal)
    world_normal[3] = 0
    world_normal = normalize(world_normal)
    return world_normal


def reflect(v: np.ndarray, n: np.ndarray) -> np.ndarray:
    return v - (2 * v.dot(n)) * n


def lighting(material: Material, light: Light, point: np.ndarray, eyev: np.ndarray, normalv: np.ndarray):
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
