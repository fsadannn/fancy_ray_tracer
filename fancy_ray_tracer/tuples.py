import numpy as np

ATOL = 1e-12
RTOL = 1e-9


class MistmatchTypeError(Exception):
    pass


def point(x: float, y: float, z: float) -> np.ndarray:
    return np.array([x, y, z, 1])


def vector(x: float, y: float, z: float) -> np.ndarray:
    return np.array([x, y, z, 0])


def equal(a: np.ndarray, b: np.ndarray) -> bool:
    return np.allclose(a, b, rtol=RTOL, atol=ATOL)


def is_vector(a: np.ndarray) -> bool:
    return a[3] < RTOL


def norm(a: np.ndarray) -> float:
    return np.linalg.norm(a)


def normalize(a: np.ndarray) -> np.ndarray:
    nm: float = np.linalg.norm(a)
    return a / nm


def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    cx = np.cross(a, b)
    return np.append(cx, 0)
