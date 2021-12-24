from math import cos, sin

import numpy as np


def identity() -> np.ndarray:
    return np.eye(4, 4)


def inverse(a: np.ndarray) -> np.ndarray:
    return np.linalg.inv(a)


def inverse_dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.solve(a, b)


def translation(x: float, y: float, z: float) -> np.ndarray:
    eye: np.ndarray = np.eye(4, 4)
    eye[0, 3] = x
    eye[1, 3] = y
    eye[2, 3] = z
    return eye


def scaling(x: float, y: float, z: float) -> np.ndarray:
    eye: np.ndarray = np.eye(4, 4)
    eye[0, 0] = x
    eye[1, 1] = y
    eye[2, 2] = z
    return eye


def rotX(rads: float) -> np.ndarray:
    eye: np.ndarray = np.eye(4, 4)
    cosr: float = cos(rads)
    sinr: float = sin(rads)
    eye[1, 1] = cosr
    eye[1, 2] = -sinr
    eye[2, 1] = sinr
    eye[2, 2] = cosr
    return eye


def rotY(rads: float) -> np.ndarray:
    eye: np.ndarray = np.eye(4, 4)
    cosr: float = cos(rads)
    sinr: float = sin(rads)
    eye[0, 0] = cosr
    eye[0, 2] = sinr
    eye[2, 0] = -sinr
    eye[2, 2] = cosr
    return eye


def rotZ(rads: float) -> np.ndarray:
    eye: np.ndarray = np.eye(4, 4)
    cosr: float = cos(rads)
    sinr: float = sin(rads)
    eye[0, 0] = cosr
    eye[0, 1] = -sinr
    eye[1, 0] = sinr
    eye[1, 1] = cosr
    return eye


# also know as skew transformation
def sharing(xy: float, xz: float, yx: float, yz: float, zx: float, zy: float) -> np.ndarray:  # pylint: disable=too-many-arguments
    eye: np.ndarray = np.eye(4, 4)
    eye[0, 1] = xy
    eye[0, 2] = xz
    eye[1, 0] = yx
    eye[1, 2] = yz
    eye[2, 0] = zx
    eye[2, 1] = zy
    return eye
