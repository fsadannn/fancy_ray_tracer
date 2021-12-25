from __future__ import annotations

from typing import Optional

import numpy as np

from .materials import make_material
from .matrices import identity, inverse
from .utils import rand_id


class Sphere:
    __slots__ = ("_id", '_transform', 'material', '_inv_transform')

    def __init__(self, sphereId: Optional[str] = None):
        self._id: str = sphereId if sphereId else rand_id()
        self._transform: np.ndarray = identity()
        self.material = make_material()
        self._inv_transform: np.ndarray = self._transform

    @property
    def transform(self) -> np.ndarray:
        return self._transform

    @property
    def inv_transform(self) -> np.ndarray:
        return self._inv_transform

    @property
    def id(self) -> str:
        return self._id

    def set_transform(self, transform: np.ndarray):
        self._transform = transform
        self._inv_transform = inverse(self._transform)

    def __eq__(self, other: Sphere) -> bool:
        if not isinstance(other, Sphere):
            raise NotImplementedError

        return self.id == other.id


def make_sphere() -> Sphere:
    return Sphere()
