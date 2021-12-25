from __future__ import annotations

from typing import Protocol

import numpy as np

from .materials import Material


class WorldObject(Protocol):
    id: str
    transform: np.ndarray
    material: Material
    inv_transform: np.ndarray

    def set_transform(self, transform: np.ndarray) -> None:
        raise NotImplementedError

    def __eq__(self, other: WorldObject) -> bool:
        raise NotImplementedError
