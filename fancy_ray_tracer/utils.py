from math import pi
from typing import List

import numpy as np

# def chain(ops: List[np.ndarray]) -> np.ndarray:
#     if len(ops) == 0:
#         return np.eye(4, 4)
#     res: np.ndarray = ops[0]
#     for op in ops[1:]:
#         res = op.dot(res)

#     return res


def chain(ops: List[np.ndarray], p: np.ndarray) -> np.ndarray:
    res: np.ndarray = p
    for op in ops:
        res = op.dot(res)

    return res
