from fancy_ray_tracer import *
from fancy_ray_tracer.constants import EPSILON


def test_intersect_miss():
    c = make_csg(CSGOperation.union, Sphere(), Cube())
    r = Ray(point(0, 2, -5), vector(0, 0, 1))
    xs = r.intersect(c)
    assert len(xs) == 0


def test_intersect():
    s1 = Sphere()
    s2 = Sphere()
    s2.set_transform(translation(0, 0, 0.5))
    c = make_csg(CSGOperation.union, s1, s2)
    r = Ray(point(0, 0, -5), vector(0, 0, 1))
    print(r.intersect(s1))
    print(r.intersect(s2))
    xs = r.intersect(c)
    assert len(xs) == 2
    assert abs(xs[0].t - 4) < EPSILON
    assert xs[0].object == s1
    assert abs(xs[1].t - 6.5) < EPSILON
    assert xs[1].object == s2
