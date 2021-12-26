from __future__ import annotations

from math import pow, sqrt

import numpy as np

from .protocols import WorldObject
from .utils import equal


class Light:
    __slots__ = ("intensity", "position")

    def __init__(self, position: np.ndarray, intensity: np.ndarray):
        self.intensity: np.ndarray = intensity
        self.position: np.ndarray = position

    def __eq__(self, other: Light) -> bool:
        if not isinstance(other, Light):
            raise NotImplementedError

        return equal(self.position, other.position) and equal(self.intensity, other.intensity)


def reflect(v: np.ndarray, n: np.ndarray) -> np.ndarray:
    return v - (2 * v.dot(n)) * n


def lighting(object: WorldObject, light: Light, point: np.ndarray,
             eyev: np.ndarray, normalv: np.ndarray, in_shadow: bool = False):
    # combine the surface color with the light's color/intensity
    material = object.material
    color = object.color_at(point)
    effective_color = color * light.intensity

    # compute the ambient contribution
    ambient = effective_color * material.ambient

    if in_shadow:
        return ambient

    # find the direction to the light source
    # normalization inplace, equivalent to
    # lightv = normalize(light.position - point)
    lightv: np.ndarray = light.position - point
    lightv *= (1.0 / sqrt(lightv.dot(lightv)))
    # light_dot_normal represents the cosine of the angle between the
    # light vector and the normal vector. A negative number means the
    # light is on the other side of the surface.
    light_dot_normal: float = lightv.dot(normalv)
    if light_dot_normal < 0:
        return ambient

    # compute the diffuse contribution
    diffuse = (material.diffuse * light_dot_normal) * effective_color
    # reflect_dot_eye represents the cosine of the angle between the
    # reflection vector and the eye vector. A negative number means the
    # light reflects away from the eye.
    # write reflect inplace for speed equivalen to
    # reflectv = reflect(-lightv, normalv)
    lightvn: np.ndarray = -lightv
    reflect_dot_eye: float = (
        lightvn - (2.0 * lightvn.dot(normalv)) * normalv).dot(eyev)

    if reflect_dot_eye <= 0:
        return ambient + diffuse

    # compute the specular contribution
    specular = light.intensity * material.specular * \
        pow(reflect_dot_eye, material.shininess)

    # Add the three contributions together to get the final shading
    return ambient + diffuse + specular
