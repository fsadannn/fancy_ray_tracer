from base64 import b64encode
from math import pi  # pylint: disable=unused-import
from os import urandom
from typing import List

import numpy as np


def chain_ops(ops: List[np.ndarray]) -> np.ndarray:
    if len(ops) == 0:
        return np.eye(4, 4)
    res: np.ndarray = ops[0]
    for op in ops[1:]:
        res = op.dot(res)

    return res


def chain(ops: List[np.ndarray], p: np.ndarray) -> np.ndarray:
    res: np.ndarray = p
    for op in ops:
        res = op.dot(res)

    return res


def rand_id(length: int = 20) -> str:
    return b64encode(urandom(length)).decode('ascii')
