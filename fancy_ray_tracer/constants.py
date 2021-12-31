from math import inf
from math import pi as _pi

import numpy as np

ATOL = 1e-5
RTOL = 1e-5
PI = _pi
EPSILON = ATOL
RAY_REFLECTION_LIMIT = 5
BOX_UNITARY_MAX_BOUND = np.array([1, 1, 1], dtype=np.float64)
BOX_UNITARY_MIN_BOUND = np.array([-1, -1, -1], dtype=np.float64)
INFINITY = inf
