import numpy as np
from fancy_ray_tracer import *
from fancy_ray_tracer.constants import EPSILON
from fancy_ray_tracer.tuples import cross


def test_deafult_triangle():
    p1 = point(0, 1, 0)
    p2 = point(-1, 0, 0)
    p3 = point(1, 0, 0)
    e1 = p2 - p1
    e2 = p3 - p1
    normal = normalize(cross(e2[:3], e1[:3]))
    t = Triangle(p1, p2, p3)
    assert equal(t.p1, p1)
    assert equal(t.p2, p2)
    assert equal(t.p3, p3)
    assert equal(t.e1, e1)
    assert equal(t.e2, e2)
    assert equal(t.normal, normal)
    assert equal(t.normal_at(np.append(np.random.rand(3), 1)), normal)


def test_intersect():
    t = Triangle(point(0, 1, 0), point(-1, 0, 0), point(1, 0, 0))

    r = Ray(point(0, -1, -2), vector(0, 1, 0))
    xs = r.intersect(t)
    assert len(xs) == 0

    r = Ray(point(1, 1, -2), vector(0, 0, 1))
    xs = r.intersect(t)
    assert len(xs) == 0

    r = Ray(point(-1, 1, -2), vector(0, 0, 1))
    xs = r.intersect(t)
    assert len(xs) == 0

    r = Ray(point(0, -1, -2), vector(0, 0, 1))
    xs = r.intersect(t)
    assert len(xs) == 0

    r = Ray(point(0, 0.5, -2), vector(0, 0, 1))
    xs = r.intersect(t)
    assert len(xs) == 1
    assert abs(xs[0].t - 2) < EPSILON
