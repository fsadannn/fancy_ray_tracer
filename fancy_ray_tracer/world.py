from typing import Iterable, List, MutableMapping, MutableSequence, Optional, Sequence

from fancy_ray_tracer.protocols import WorldObject
from fancy_ray_tracer.ray import Intersection, Ray

from .ilumination import Light


class World:
    __slots__ = ("light", 'objects', '_objects_ids')

    def __init__(self, light: Optional[Light] = None, objects: MutableSequence[WorldObject] = []) -> None:
        self.light: Light = light
        self.objects: MutableSequence[WorldObject] = objects
        self._objects_ids: MutableMapping[str, WorldObject] = {
            i.id: n for n, i in enumerate(objects)}

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

        intersections.sort()
        return intersections
