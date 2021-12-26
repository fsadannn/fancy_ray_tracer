from base64 import b64encode
from os import urandom
from typing import Sequence, Tuple, Union

import numpy as np

from .constants import ATOL, RTOL


def chain_ops(ops: Sequence[np.ndarray]) -> np.ndarray:
    if len(ops) == 0:
        return np.eye(4, 4)
    res: np.ndarray = ops[-1]
    for op in ops[-2::-1]:
        res = op.dot(res)

    return res


def chain(ops: Sequence[np.ndarray], p: np.ndarray) -> np.ndarray:
    res: np.ndarray = p
    for op in ops:
        res = op.dot(res)

    return res


def rand_id(length: int = 20) -> str:
    return b64encode(urandom(length)).decode('ascii')


def equal(a: np.ndarray, b: np.ndarray, atol=ATOL, rtol=RTOL) -> bool:
    return np.allclose(a, b, rtol=atol, atol=rtol)


def is_vector(a: np.ndarray) -> bool:
    return a[3] < ATOL


def colorf_to_color(color: Union[np.ndarray, Tuple[float, float, float]]) -> Tuple[int, int, int]:
    new_color = (min(int(color[0] * 255), 255),
                 min(int(color[1] * 255), 255), min(int(color[2] * 255), 255))
    return new_color
