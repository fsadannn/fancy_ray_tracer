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

from fancy_ray_tracer.tuples import make_color

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
            return lighting(cmp.object.material, self.light[0],
                            cmp.point, cmp.eyev, cmp.normalv)

        if len(self.light) == 0:
            return make_color(0, 0, 0)

        color = lighting(cmp.object.material, self.light[0],
                         cmp.point, cmp.eyev, cmp.normalv)
        light: Light
        for light in self.light[1:]:
            color += lighting(cmp.object.material, light,
                              cmp.point, cmp.eyev, cmp.normalv)

        return color

    def color_at(self, ray: Ray) -> np.ndarray:
        intersections: Sequence[Intersection] = self.intersec(ray)
        it: Optional[Intersection] = hit_sorted(intersections)

        if it is None:
            return make_color(0, 0, 0)

        return self.shade_hit(Computations(it, ray))
