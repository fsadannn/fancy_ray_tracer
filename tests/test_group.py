from math import sqrt

from fancy_ray_tracer import *
from fancy_ray_tracer.constants import PI
from fancy_ray_tracer.ray import normal_to_world, world_to_object


def test_intersec_empty():
    g = Group()
    r = Ray(point(0, 0, 0), vector(0, 0, 1))
    xs = r.intersect(g)
    assert len(xs) == 0


def test_intersec():
    g = Group()
    s1 = Sphere()
    s2 = Sphere()
    s2.set_transform(translation(0, 0, -3))
    s3 = Sphere()
    s3.set_transform(translation(5, 0, 0))
    g.add_shape(s1)
    g.add_shape(s2)
    g.add_shape(s3)
    r = Ray(point(0, 0, -5), vector(0, 0, 1))
    xs = r.intersect(g)
    assert len(xs) == 4
    assert xs[0].object == s2
    assert xs[1].object == s2
    assert xs[2].object == s1
    assert xs[3].object == s1


def test_intersec_transform():
    g = Group()
    g.set_transform(scaling(2, 2, 2))
    s = Sphere()
    s.set_transform(translation(5, 0, 0))
    g.add_shape(s)
    r = Ray(point(10, 0, -10), vector(0, 0, 1))
    xs = r.intersect(g)
    assert len(xs) == 2


def test_normal_world_to_obj():
    g1 = Group()
    g1.set_transform(rotY(PI / 2))
    g2 = Group()
    g2.set_transform(scaling(2, 2, 2))
    g1.add_shape(g2)
    s = Sphere()
    s.set_transform(translation(5, 0, 0))
    g2.add_shape(s)
    p = world_to_object(s, point(-2, 0, -10))
    assert equal(p, point(0, 0, -1))


def test_normal_obj_to_world():
    g1 = Group()
    g1.set_transform(rotY(PI / 2))
    g2 = Group()
    g2.set_transform(scaling(1, 2, 3))
    g1.add_shape(g2)
    s = Sphere()
    s.set_transform(translation(5, 0, 0))
    g2.add_shape(s)
    n = normal_to_world(s, vector(sqrt(3) / 3, sqrt(3) / 3, sqrt(3) / 3))
    assert equal(n, vector(0.2857, 0.4286, -0.8571), atol=1e-4, rtol=1e-4)


def test_normal():
    g1 = Group()
    g1.set_transform(rotY(PI / 2))
    g2 = Group()
    g2.set_transform(scaling(1, 2, 3))
    g1.add_shape(g2)
    s = Sphere()
    s.set_transform(translation(5, 0, 0))
    g2.add_shape(s)
    n = normal_at(s, point(1.7321, 1.1547, -5.5774))
    assert equal(n, vector(0.2857, 0.4286, -0.8571), atol=1e-4, rtol=1e-4)
