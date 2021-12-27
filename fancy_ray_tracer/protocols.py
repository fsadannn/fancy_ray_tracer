from __future__ import annotations

from abc import abstractmethod
from os import PathLike
from typing import Any, Optional, Protocol, Sequence, Tuple, Union

import numpy as np

from .matrices import inverse

ColorInput = Union[
    Sequence[int], Tuple[int, int, int], Tuple[int, int, int, int]
]

ColorOutput = Sequence[int]


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


class CanvasP(Protocol):
    _screenSize: Tuple[int, int]
    _canvas: Any

    @property
    def screen_size(self):
        return self._screenSize

    @property
    def width(self) -> int:
        return self._screenSize[0]

    @property
    def height(self) -> int:
        return self._screenSize[1]

    @property
    def canvas(self) -> Any:
        return self._canvas

    @abstractmethod
    def set_pixel(self, x: int, y: int, color: ColorInput):
        raise NotImplementedError

    def set_pixelf(self, x: int, y: int, color: Union[Tuple[float, float, float], np.ndarray]):
        new_color = (min(int(color[0] * 255), 255),
                     min(int(color[1] * 255), 255), min(int(color[2] * 255), 255))
        self.set_pixel(x, y, new_color)

    @abstractmethod
    def get_pixel(self, x: int, y: int) -> ColorOutput:
        raise NotImplementedError

    @abstractmethod
    def save_img(self, file: Union[str, PathLike]):
        raise NotImplementedError
