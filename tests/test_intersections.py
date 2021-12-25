from fancy_ray_tracer import Intersection, Ray, make_sphere, point, scaling, vector
from fancy_ray_tracer.constants import ATOL


def test_intersect():
    ray = Ray(point(0, 0, -5), vector(0, 0, 1))
    sphere = make_sphere()
    xs = ray.intersect(sphere)
    assert len(xs) == 2
    assert abs(xs[0].t - 4) < ATOL
    assert abs(xs[1].t - 6) < ATOL


def test_intersect_tanget():
    ray = Ray(point(0, 1, -5), vector(0, 0, 1))
    sphere = make_sphere()
    xs = ray.intersect(sphere)
    assert len(xs) == 2
    assert abs(xs[0].t - 5) < ATOL
    assert abs(xs[1].t - 5) < ATOL


def test_intersect_miss():
    ray = Ray(point(0, 2, -5), vector(0, 0, 1))
    sphere = make_sphere()
    xs = ray.intersect(sphere)
    assert len(xs) == 0


def test_intersect_inside():
    ray = Ray(point(0, 0, 0), vector(0, 0, 1))
    sphere = make_sphere()
    xs = ray.intersect(sphere)
    assert len(xs) == 2
    assert abs(xs[0].t + 1) < ATOL
    assert abs(xs[1].t - 1) < ATOL


def test_intersect_behind():
    ray = Ray(point(0, 0, 5), vector(0, 0, 1))
    sphere = make_sphere()
    xs = ray.intersect(sphere)
    assert len(xs) == 2
    assert abs(xs[0].t + 6) < ATOL
    assert abs(xs[1].t + 4) < ATOL


def test_intersection_class():
    sphere = make_sphere()
    inter = Intersection(3.5, sphere)
    assert inter.object.id == sphere.id
    assert abs(inter.t - 3.5) < ATOL


def test_intersect_sphere():
    r = Ray(point(0, 0, -5), vector(0, 0, 1))
    s = make_sphere()
    s.set_transform(scaling(2, 2, 2))
    xs = r.intersect(s)
    assert len(xs) == 2
    assert abs(xs[0].t - 3) < ATOL
    assert abs(xs[1].t - 7) < ATOL
