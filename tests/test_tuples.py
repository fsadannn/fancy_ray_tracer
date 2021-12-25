import numpy as np
from fancy_ray_tracer import equal, tuples
from fancy_ray_tracer.constants import ATOL, RTOL


def test_make_point():
    pt = tuples.point(4, -4, 3)
    assert np.allclose(pt, np.array(
        [4, -4, 3, 1]), rtol=RTOL, atol=ATOL)


def test_make_vector():
    vc = tuples.vector(4, -4, 3)
    assert np.allclose(vc, np.array(
        [4, -4, 3, 0]), rtol=RTOL, atol=ATOL)


def test_equal():
    vc = tuples.vector(4, -4, 3)
    assert equal(vc, vc)
