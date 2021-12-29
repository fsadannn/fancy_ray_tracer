from math import sqrt
from typing import (
    Iterable,
    List,
    MutableMapping,
    MutableSequence,
    Optional,
    Sequence,
    Union,
)

import numpy as np

try:
    from .compiled import _schlick
except ImportError:
    _schlick = None
from .constants import EPSILON, RAY_REFLECTION_LIMIT
from .illumination import Light, lighting
from .protocols import WorldObject
from .ray import Computations, Intersection, Ray, hit_sorted
from .tuples import make_color

_BLACK = make_color(0, 0, 0)
_WHITE = make_color(1, 1, 1)


class World:
    __slots__ = ("light", 'objects', '_objects_ids')

    def __init__(self, light: Union[Light, Iterable[Light]] = (),
                 objects: Iterable[WorldObject] = ()) -> None:
        if isinstance(light, Iterable):
            self.light: MutableSequence[Light] = list(light)
        else:
            self.light = [light]
        self.objects: MutableSequence[WorldObject] = list(objects)
        self._objects_ids: MutableMapping[str, WorldObject] = {
            i.id: n for n, i in enumerate(self.objects)}

    def add_light(self, light: Light):
        self.light.append(light)

    def add_object(self, obj: WorldObject) -> None:
        self.objects.append(obj)

    def add_objects(self, objs: Iterable[WorldObject]) -> None:
        objects = self.objects
        for obj in objs:
            self._objects_ids[obj.id] = len(objects)
            objects.append(obj)

    def has_object(self, obj: WorldObject):
        return obj.id in self._objects_ids

    def has_object_id(self, obj_id: str):
        return obj_id in self._objects_ids

    def intersec(self, ray: Ray) -> Sequence[Intersection]:
        intersections: List[Intersection] = []
        obj: WorldObject
        for obj in self.objects:
            intersects = ray.intersect(obj)
            if len(intersects) != 0:
                intersections.extend(intersects)

        if len(intersections) == 0:
            return intersections

        intersections.sort()
        return intersections

    def shade_hit(self, cmp: Computations, remaining: int = RAY_REFLECTION_LIMIT) -> np.ndarray:
        material = cmp.object.material
        if len(self.light) == 1:
            in_shadow = self._is_shadowed(
                cmp.over_point, self.light[0].position)
            surface = lighting(cmp.object, self.light[0],
                               cmp.over_point, cmp.eyev, cmp.normalv, in_shadow)
            reflected = self.reflected_color(cmp, remaining)
            refracted = self.refracted_color(cmp, remaining)

            if material.reflective > EPSILON and material.transparency > EPSILON:
                # reflectance = schlick(cmp)
                reflectance: float = schlick(
                    cmp.eyev, cmp.normalv, cmp.n1, cmp.n2)
                return surface + reflected * reflectance + (1 - reflectance) * refracted

            return surface + reflected + refracted

        if len(self.light) == 0:
            return _BLACK

        reflected = self.reflected_color(cmp, remaining)
        refracted = self.refracted_color(cmp, remaining)
        in_shadow = self._is_shadowed(cmp.over_point, self.light[0].position)
        color = lighting(cmp.object, self.light[0],
                         cmp.over_point, cmp.eyev, cmp.normalv, in_shadow)
        light: Light
        for light in self.light[1:]:
            in_shadow = self._is_shadowed(cmp.over_point, light.position)
            color += lighting(cmp.object, light,
                              cmp.over_point, cmp.eyev, cmp.normalv, in_shadow)

        if material.reflective > EPSILON and material.transparency > EPSILON:
            # reflectance = schlick(cmp)
            reflectance = schlick(
                cmp.eyev, cmp.normalv, cmp.n1, cmp.n2)
            return color + reflected * reflectance + (1 - reflectance) * refracted

        return color + reflected + refracted

    def color_at(self, ray: Ray, remaining: int = RAY_REFLECTION_LIMIT) -> np.ndarray:
        intersections: Sequence[Intersection] = self.intersec(ray)
        it: Optional[Intersection] = hit_sorted(intersections)

        if it is None:
            return _BLACK

        return self.shade_hit(Computations(it, ray, intersections), remaining)

    def is_shadowed(self, p: np.ndarray) -> bool:
        if len(self.light) == 1:
            return self._is_shadowed(p, self.light[0].position)

        if len(self.light) == 0:
            return True

        in_shadow: bool = self._is_shadowed(p, self.light[0].position)
        light: Light
        for light in self.light[1:]:
            in_shadow |= self._is_shadowed(p, light.position)

        return in_shadow

    def _is_shadowed(self, p: np.ndarray, light_position: np.ndarray) -> bool:
        direction: np.ndarray = light_position - p
        distance: float = sqrt(direction.dot(direction))
        direction *= (1 / distance)

        r = Ray(p, direction)
        intersects = self.intersec(r)
        h = hit_sorted(intersects)

        return h is not None and h.t < distance

    def reflected_color(self, cmp: Computations, remaining: int = RAY_REFLECTION_LIMIT) -> np.ndarray:
        if cmp.object.material.reflective < EPSILON or remaining <= 0:
            return _BLACK

        reflect_ray = Ray(cmp.over_point, cmp.reflectv)
        color = self.color_at(reflect_ray, remaining - 1)
        return color * cmp.object.material.reflective

    def refracted_color(self, cmp: Computations, remaining: int = RAY_REFLECTION_LIMIT) -> np.ndarray:
        if cmp.object.material.transparency < EPSILON or remaining <= 0:
            return _BLACK

        n_ratio = cmp.n1 / cmp.n2
        cos_i: float = cmp.eyev.dot(cmp.normalv)
        sin2_t = n_ratio**2 * (1.0 - cos_i**2)

        if sin2_t > 1:
            return _BLACK

        cos_t = sqrt(1.0 - sin2_t)
        direction = cmp.normalv * \
            (n_ratio * cos_i - cos_t) - cmp.eyev * n_ratio
        refract_ray = Ray(cmp.under_point, direction)

        return self.color_at(refract_ray, remaining - 1) * cmp.object.material.transparency


def schlick_fallback(eyev: np.ndarray, normalv: np.ndarray, n1: float, n2: float) -> float:  # reflectance
    cos: float = eyev.dot(normalv)

    if n1 > n2:
        sin2_t = (n1 / n2)**2 * (1 - cos**2)

        if sin2_t > 1:
            return 1.0

        cos = sqrt(1.0 - sin2_t)

    r0 = ((n1 - n2) / (n1 + n2))**2
    return r0 + (1 - r0) * (1 - cos)**5


schlick = _schlick.schlick if _schlick is not None else schlick_fallback
