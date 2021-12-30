import numpy as np
from fancy_ray_tracer import equal, tuples
from fancy_ray_tracer.constants import ATOL, RTOL


def test_add_color():
    c1 = tuples.make_color(0.9, 0.6, 0.75)
    c2 = tuples.make_color(0.7, 0.1, 0.25)
    cp = c1 + c2
    assert equal(cp, tuples.make_color(1.6, 0.7, 1.0))


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
