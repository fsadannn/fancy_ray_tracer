from math import inf
from math import pi as _pi

import numpy as np

ATOL: float = 1e-5
RTOL: float = 1e-5
PI: float = _pi
EPSILON: float = ATOL
RAY_REFLECTION_LIMIT: int = 5
BOX_UNITARY_MAX_BOUND: np.ndarray = np.array((1, 1, 1, 1), dtype=np.float64)
BOX_UNITARY_MIN_BOUND: np.ndarray = np.array((-1, -1, -1, 1), dtype=np.float64)
INFINITY: float = inf
