from __future__ import annotations

from typing import Optional, Protocol, Sequence

import numpy as np

from .matrices import inverse


class Transformable(Protocol):
    transform: np.ndarray
    inv_transform: np.ndarray

    def set_transform(self, transform: np.ndarray) -> None:
        self.transform = transform
        self.inv_transform = inverse(self.transform)


class ColorAtPoint(Protocol):
    def color_at(self, point: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Pattern(Transformable, ColorAtPoint, Protocol):
    name: str

    def __eq__(self, other: Pattern) -> bool:
        raise NotImplementedError


class MaterialP(ColorAtPoint, Protocol):
    color: np.ndarray
    ambient: float
    diffuse: float
    specular: float
    shininess: float
    pattern: Optional[Pattern]
    reflective: float
    transparency: float
    refractive_index: float

    def __eq__(self, other: MaterialP) -> bool:
        raise NotImplementedError


class IntersectionP(Protocol):
    t: float
    object: WorldObject


class WorldObject(Transformable, ColorAtPoint, Protocol):
    id: str
    material: MaterialP

    def __eq__(self, other: WorldObject) -> bool:
        raise NotImplementedError

    def normal_at(self, p: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def intersect(self, origin: np.ndarray, direction: np.ndarray) -> Sequence[IntersectionP]:
        raise NotImplementedError
