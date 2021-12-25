from __future__ import annotations

from typing import Protocol

import numpy as np

from .materials import Material


class WorldObject(Protocol):
    _id: str
    _transform: np.ndarray
    material: Material
    _inv_transform: np.ndarray

    @property
    def transform(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def inv_transform(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def id(self) -> str:
        raise NotImplementedError

    def set_transform(self, transform: np.ndarray) -> None:
        raise NotImplementedError

    def __eq__(self, other: WorldObject) -> bool:
        raise NotImplementedError
