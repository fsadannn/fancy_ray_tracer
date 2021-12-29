from fancy_ray_tracer import *
from fancy_ray_tracer.constants import EPSILON


def test_intersect():
    c = Cube()

    origins = [
        point(5, 0.5, 0),  # +x
        point(-5, 0.5, 0),  # -x
        point(0.5, 5, 0),  # +y
        point(0.5, -5, 0),  # -x
        point(0.5, 0, 5),  # +z
        point(0.5, 0, -5),  # -z
        point(0, 0.5, 0),  # inside
    ]
    directions = [
        vector(-1, 0, 0),  # +x
        vector(1, 0, 0),  # -x
        vector(0, -1, 0),  # +y
        vector(0, 1, 0),  # -x
        vector(0, 0, -1),  # +z
        vector(0, 0, 1),  # -z
        vector(0, 0, 1),  # inside
    ]
    t1s = [4, 4, 4, 4, 4, 4, -1]
    t2s = [6, 6, 6, 6, 6, 6, 1]
    for origin, direction, t1, t2 in zip(origins, directions, t1s, t2s):
        its = c.intersect(origin, direction)
        assert len(its) == 2
        assert abs(its[0].t - t1) < EPSILON
        assert abs(its[1].t - t2) < EPSILON


def test_miss():
    c = Cube()

    origins = [
        point(-2, 0, 0),
        point(0, -2, 0),
        point(0, 0, -2),
        point(2, 0, 2),
        point(0, 2, 2, ),
        point(2, 2, 0),
    ]
    directions = [
        vector(0.2673, 0.5345, 0.8018),
        vector(0.8018, 0.2673, 0.5345),
        vector(0.5345, 0.8018, 0.2673),
        vector(0, 0, -1),
        vector(0, -1, 0),
        vector(-1, 0, 0),
    ]
    for origin, direction in zip(origins, directions):
        its = c.intersect(origin, direction)
        assert len(its) == 0


def test_normals():
    c = Cube()
    points = [
        point(1, 0.5, -0.8),
        point(-1, -0.2, 0.9),
        point(-0.4, 1, -0.1),
        point(0.3, -1, -0.7),
        point(-0.6, 0.3, 1),
        point(0.4, 0.4, -1),
        point(1, 1, 1),
        point(-1, -1, -1)
    ]

    normals = [
        vector(1, 0, 0),
        vector(-1, 0, 0),
        vector(0, 1, 0),
        vector(0, -1, 0),
        vector(0, 0, 1),
        vector(0, 0, -1),
        vector(1, 0, 0),
        vector(-1, 0, 0),
    ]

    for p, n in zip(points, normals):
        nm = c.normal_at(p)
        assert equal(nm, n)
