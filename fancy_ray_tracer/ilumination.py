from __future__ import annotations

from math import pow, sqrt

import numpy as np

from .materials import Material
# from .tuples import normalize
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


def lighting(material: Material, light: Light, point: np.ndarray,
             eyev: np.ndarray, normalv: np.ndarray):
    # combine the surface color with the light's color/intensity
    effective_color = material.color * light.intensity

    # find the direction to the light source
    # normalization inplace, equivalent to
    # lightv = normalize(light.position - point)
    lightv: np.ndarray = light.position - point
    nm: float = sqrt(lightv.dot(lightv))
    lightv *= (1.0 / nm)
    # compute the ambient contribution
    ambient = effective_color * material.ambient
    # light_dot_normal represents the cosine of the angle between the
    # light vector and the normal vector. A negative number means the
    # light is on the other side of the surface.
    light_dot_normal: float = lightv.dot(normalv)
    if light_dot_normal < 0:
        # diffuse = 0
        # specular = 0
        return ambient
    # else:

    # compute the diffuse contribution
    diffuse = (material.diffuse * light_dot_normal) * effective_color
    # reflect_dot_eye represents the cosine of the angle between the
    # reflection vector and the eye vector. A negative number means the
    # light reflects away from the eye.
    # write reflect inplace for speed equivalen to
    # reflectv = reflect(-lightv, normalv)
    lightvn: np.ndarray = -lightv
    reflectv: np.ndarray = lightvn - (2.0 * lightvn.dot(normalv)) * normalv

    reflect_dot_eye: float = reflectv.dot(eyev)

    if reflect_dot_eye <= 0:
        # specular = 0
        return ambient + diffuse
    # else:

    # compute the specular contribution
    factor = pow(reflect_dot_eye, material.shininess)
    specular = light.intensity * material.specular * factor

    # Add the three contributions together to get the final shading
    return ambient + diffuse + specular
