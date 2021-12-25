from math import sqrt

import numpy as np


class MistmatchTypeError(Exception):
    pass


def make_color(r: float, g: float, b: float) -> np.ndarray:
    return np.array([r, g, b], dtype=np.float64)


def point(x: float, y: float, z: float) -> np.ndarray:
    return np.array([x, y, z, 1], dtype=np.float64)


def vector(x: float, y: float, z: float) -> np.ndarray:
    return np.array([x, y, z, 0], dtype=np.float64)


def normalize(x: np.ndarray) -> np.ndarray:
    #nm: float = np.linalg.norm(a)
    # return a / nm
    nm = sqrt(x.dot(x))
    return x * (1.0 / nm)


def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    cx = np.cross(a, b)
    return np.append(cx, 0.0)
