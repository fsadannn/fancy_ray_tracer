from __future__ import annotations

from typing import Optional

import numpy as np

from .materials import make_material
from .matrices import identity, inverse
from .utils import rand_id


class Sphere:
    __slots__ = ("id", 'transform', 'material', 'inv_transform')

    def __init__(self, sphereId: Optional[str] = None):
        self.id: str = sphereId if sphereId else rand_id()
        self.transform: np.ndarray = identity()
        self.material = make_material()
        self.inv_transform: np.ndarray = self.transform

    def set_transform(self, transform: np.ndarray):
        self.transform = transform
        self.inv_transform = inverse(self.transform)

    def __eq__(self, other: Sphere) -> bool:
        if not isinstance(other, Sphere):
            raise NotImplementedError

        return self.id == other.id


def make_sphere() -> Sphere:
    return Sphere()
