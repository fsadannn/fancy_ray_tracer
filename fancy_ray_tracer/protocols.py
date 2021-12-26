from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from .materials import Material
from .utils import rand_id


class IntersectionP(Protocol):
    t: float
    object: WorldObject


class WorldObject(Protocol):
    id: str = rand_id()
    transform: np.ndarray
    material: Material
    inv_transform: np.ndarray

    def set_transform(self, transform: np.ndarray) -> None:
        raise NotImplementedError

    def __eq__(self, other: WorldObject) -> bool:
        raise NotImplementedError

    def normal_at(self, p: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def intersect(self, origin: np.ndarray, direction: np.ndarray) -> Sequence[IntersectionP]:
        raise NotImplementedError
