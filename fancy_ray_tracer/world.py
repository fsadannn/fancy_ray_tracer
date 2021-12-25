

from typing import Iterable, MutableMapping, MutableSequence, Optional

from fancy_ray_tracer.protocols import WorldObject

from .ilumination import Light


class World:
    __slots__ = ("light", '_objects', '_objects_ids')

    def __init__(self, light: Optional[Light] = None, objects: MutableSequence[WorldObject] = []) -> None:
        self.light: Light = light
        self._objects: MutableSequence[WorldObject] = objects
        self._objects_ids: MutableMapping[str, WorldObject] = {
            i.id: n for n, i in enumerate(objects)}

    @property
    def objects(self):
        return self._objects

    def add_object(self, obj: WorldObject) -> None:
        self._objects.append(obj)

    def add_objects(self, objs: Iterable[WorldObject]) -> None:
        objects = self._objects
        for obj in objs:
            self._objects_ids[obj.id] = len(objects)
            objects.append(obj)

    def has_object(self, obj: WorldObject):
        return obj.id in self._objects_ids

    def has_object_id(self, obj_id: str):
        return obj_id in self._objects_ids
