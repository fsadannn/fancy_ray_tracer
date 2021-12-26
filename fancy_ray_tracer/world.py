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

from fancy_ray_tracer.tuples import make_color, point

from .illumination import Light, lighting
from .protocols import WorldObject
from .ray import Computations, Intersection, Ray, hit_sorted


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
            if intersects:
                intersections.extend(intersects)

        if len(intersections) == 0:
            return intersections

        intersections.sort()
        return intersections

    def shade_hit(self, cmp: Computations) -> np.ndarray:
        if len(self.light) == 1:
            in_shadow = self._is_shadowed(
                cmp.over_point, self.light[0].position)
            return lighting(cmp.object, self.light[0],
                            cmp.over_point, cmp.eyev, cmp.normalv, in_shadow)

        if len(self.light) == 0:
            return make_color(0, 0, 0)

        in_shadow = self._is_shadowed(cmp.over_point, self.light[0].position)
        color = lighting(cmp.object, self.light[0],
                         cmp.over_point, cmp.eyev, cmp.normalv, in_shadow)
        light: Light
        for light in self.light[1:]:
            in_shadow = self._is_shadowed(cmp.over_point, light.position)
            color += lighting(cmp.object, light,
                              cmp.over_point, cmp.eyev, cmp.normalv, in_shadow)

        return color

    def color_at(self, ray: Ray) -> np.ndarray:
        intersections: Sequence[Intersection] = self.intersec(ray)
        it: Optional[Intersection] = hit_sorted(intersections)

        if it is None:
            return make_color(0, 0, 0)

        return self.shade_hit(Computations(it, ray))

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
