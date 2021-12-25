import numpy as np
from fancy_ray_tracer.matrices import scaling, translation, view_transform
from fancy_ray_tracer.tuples import point, vector
from fancy_ray_tracer.utils import equal


def test_view_transform():
    fromp = point(0, 0, 0)
    to = point(0, 0, -1)
    up = vector(0, 1, 0)
    t = view_transform(fromp, to, up)
    assert equal(t, np.eye(4, 4))


def test_view_transform_plus_z():
    fromp = point(0, 0, 0)
    to = point(0, 0, 1)
    up = vector(0, 1, 0)
    t = view_transform(fromp, to, up)
    assert equal(t, scaling(-1, 1, -1))


def test_view_transform_2():
    fromp = point(0, 0, 8)
    to = point(0, 0, 0)
    up = vector(0, 1, 0)
    t = view_transform(fromp, to, up)
    assert equal(t, translation(0, 0, -8))


def test_view_transform_3():
    fromp = point(1, 3, 2)
    to = point(4, -2, 8)
    up = vector(1, 1, 0)
    t = view_transform(fromp, to, up)
    tt = np.array([[-0.50709, 0.50709, 0.67612, -2.36643],
                   [0.76772, 0.60609, 0.12122, -2.82843],
                   [-0.35857, 0.59761, -0.71714, 0.00000],
                   [0.00000, 0.00000, 0.00000, 1.00000]])
    assert equal(t, tt)
