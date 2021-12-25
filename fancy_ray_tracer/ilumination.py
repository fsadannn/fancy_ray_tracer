from __future__ import annotations

import numpy as np

from .materials import Material
from .tuples import normalize
from .utils import equal


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
