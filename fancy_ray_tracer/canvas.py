from os import PathLike
from typing import Tuple, Union

from PIL.Image import Image
from PIL.Image import new as newImage
from PIL.PyAccess import PyAccess

from .protocols import CanvasP, ColorInput, ColorOutput


class Canvas(CanvasP):
    def __init__(self, screenSize: Tuple[int, int] = (512, 512)):
        self._screenSize: Tuple[int, int] = tuple(screenSize)
        self._canvas: Image = newImage(
            mode="RGB", size=self._screenSize, color=(0, 0, 0))
        self._pixels: PyAccess = self._canvas.load()

    def get_pixel(self, x: int, y: int) -> ColorOutput:
        return self._canvas[x, y]

    def set_pixel(self, x: int, y: int, color: ColorInput):
        self._pixels[x, y] = color

    def save_img(self, file: Union[str, PathLike]):
        self._canvas.save(file)
