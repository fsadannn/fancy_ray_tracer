from math import sqrt

from fancy_ray_tracer import *
from fancy_ray_tracer.constants import EPSILON


def test_intersect():
    c = Cone()

    origins = [
        point(0, 0, -5),
        point(0, 0, -5),
        point(1, 1, -5),
    ]
    directions = [
        vector(0, 0, 1),
        vector(1, 1, 1),
        vector(-0.5, -1, 1),
    ]
    t1s = [5, 8.66025, 4.55006]
    t2s = [5, 8.66025, 49.44994]
    for origin, direction, t1, t2 in zip(origins, directions, t1s, t2s):
        its = c.intersect(origin, normalize(direction))
        assert len(its) == 2
        assert abs(its[0].t - t1) < EPSILON
        assert abs(its[1].t - t2) < EPSILON

    xs = c.intersect(point(0, 0, -1), normalize(vector(0, 1, 1)))
    assert len(xs) == 1
    assert abs(xs[0].t - 0.35355) < EPSILON


def test_closed():
    c = Cone(-0.5, 0.5, True)

    origins = [
        point(0, 0, -5),
        point(0, 0, -0.25),
        point(0, 0, -0.25),
    ]
    directions = [
        vector(0, 1, 0),
        vector(1, 1, 1),
        vector(0, 1, 0),
    ]
    counts = [0, 2, 4]

    for origin, direction, count in zip(origins, directions, counts):
        its = c.intersect(origin, normalize(direction))
        assert len(its) == count


def test_normals():
    c = Cone()
    points = [
        point(0, 0, 0),
        point(1, 1, 1),
        point(-1, -1, 0),
    ]

    normals = [
        vector(0, 0, 0),
        vector(1, -sqrt(2), 1),
        vector(-1, 1, 0),
    ]

    for p, n in zip(points, normals):
        nm = c.normal_at(p)
        assert equal(nm, n)
