from math import sqrt

from fancy_ray_tracer import Intersection, Ray, Sphere, point, scaling, vector
from fancy_ray_tracer.constants import ATOL, EPSILON
from fancy_ray_tracer.matrices import translation
from fancy_ray_tracer.primitives import Plane, glass_sphere
from fancy_ray_tracer.ray import Computations
from fancy_ray_tracer.utils import equal
from fancy_ray_tracer.world import schlick


def test_intersect():
    ray = Ray(point(0, 0, -5), vector(0, 0, 1))
    sphere = Sphere()
    xs = ray.intersect(sphere)
    assert len(xs) == 2
    assert abs(xs[0].t - 4) < ATOL
    assert abs(xs[1].t - 6) < ATOL


def test_intersect_tanget():
    ray = Ray(point(0, 1, -5), vector(0, 0, 1))
    sphere = Sphere()
    xs = ray.intersect(sphere)
    assert len(xs) == 2
    assert abs(xs[0].t - 5) < ATOL
    assert abs(xs[1].t - 5) < ATOL


def test_intersect_miss():
    ray = Ray(point(0, 2, -5), vector(0, 0, 1))
    sphere = Sphere()
    xs = ray.intersect(sphere)
    assert len(xs) == 0


def test_intersect_inside():
    ray = Ray(point(0, 0, 0), vector(0, 0, 1))
    sphere = Sphere()
    xs = ray.intersect(sphere)
    assert len(xs) == 2
    assert abs(xs[0].t + 1) < ATOL
    assert abs(xs[1].t - 1) < ATOL


def test_intersect_behind():
    ray = Ray(point(0, 0, 5), vector(0, 0, 1))
    sphere = Sphere()
    xs = ray.intersect(sphere)
    assert len(xs) == 2
    assert abs(xs[0].t + 6) < ATOL
    assert abs(xs[1].t + 4) < ATOL


def test_intersection_class():
    sphere = Sphere()
    inter = Intersection(3.5, sphere)
    assert inter.object.id == sphere.id
    assert abs(inter.t - 3.5) < ATOL


def test_intersect_sphere():
    r = Ray(point(0, 0, -5), vector(0, 0, 1))
    s = Sphere()
    s.set_transform(scaling(2, 2, 2))
    xs = r.intersect(s)
    assert len(xs) == 2
    assert abs(xs[0].t - 3) < ATOL
    assert abs(xs[1].t - 7) < ATOL


def test_computations():
    r = Ray(point(0, 0, -5), vector(0, 0, 1))
    s = Sphere()
    i = Intersection(4, s)
    comps = Computations(i, r)
    assert comps.t == i.t
    assert comps.object == i.object
    assert equal(comps.point, point(0, 0, -1))
    assert equal(comps.eyev, vector(0, 0, -1))
    assert equal(comps.normalv, vector(0, 0, -1))


def test_hit_outside():
    r = Ray(point(0, 0, -5), vector(0, 0, 1))
    s = Sphere()
    i = Intersection(4, s)
    comps = Computations(i, r)
    assert comps.inside == False


def test_hit_inside():
    r = Ray(point(0, 0, 0), vector(0, 0, 1))
    s = Sphere()
    i = Intersection(1, s)
    comps = Computations(i, r)
    assert comps.inside == True
    assert equal(comps.point, point(0, 0, 1))
    assert equal(comps.eyev, vector(0, 0, -1))
    assert equal(comps.normalv, vector(0, 0, -1))


def test_hit_offset_point():
    r = Ray(point(0, 0, -5), vector(0, 0, 1))
    s = Sphere()
    s.set_transform(translation(0, 0, 1))
    i = Intersection(5, s)
    comps = Computations(i, r)
    assert comps.over_point[2] < -ATOL / 2
    assert comps.point[2] > comps.over_point[2]


def test_hit_offset_point2():
    r = Ray(point(0, 0, -5), vector(0, 0, 1))
    s = Sphere()
    s.set_transform(translation(0, 0, 1))
    i = Intersection(5, s)
    comps = Computations(i, r)
    assert comps.under_point[2] > ATOL / 2
    assert comps.point[2] < comps.under_point[2]


def test_plane_ray_parallel():
    p = Plane()
    r = Ray(point(0, 10, 0), vector(0, 0, 1))
    xs = r.intersect(p)
    assert len(xs) == 0


def test_plane_ray_coplanar():
    p = Plane()
    r = Ray(point(0, 0, 0), vector(0, 0, 1))
    xs = r.intersect(p)
    assert len(xs) == 0


def test_plane_ray_above():
    p = Plane()
    r = Ray(point(0, 1, 0), vector(0, -1, 0))
    xs = r.intersect(p)
    assert len(xs) == 1
    assert abs(xs[0].t - 1) < ATOL
    assert xs[0].object == p


def test_plane_ray_below():
    p = Plane()
    r = Ray(point(0, -1, 0), vector(0, 1, 0))
    xs = r.intersect(p)
    assert len(xs) == 1
    assert abs(xs[0].t - 1) < ATOL
    assert xs[0].object == p


def test_reflection():
    p = Plane()
    r = Ray(point(0, 1, -1), vector(0, -sqrt(2) / 2, sqrt(2) / 2))
    i = Intersection(sqrt(2), p)
    cmp = Computations(i, r)
    assert equal(cmp.reflectv, vector(0, sqrt(2) / 2, sqrt(2) / 2))


def test_refraction_idex():
    A = glass_sphere()
    A.set_transform(scaling(2, 2, 2))
    A.material.refractive_index = 1.5
    B = glass_sphere()
    B.set_transform(translation(0, 0, -0.25))
    B.material.refractive_index = 2.0
    C = glass_sphere()
    C.set_transform(translation(0, 0, 0.25))
    C.material.refractive_index = 2.5
    r = Ray(point(0, 0, -4), vector(0, 0, 1))
    xs = [Intersection(2, A), Intersection(2.75, B), Intersection(
        3.25, C), Intersection(4.75, B), Intersection(5.25, C), Intersection(6, A)]
    comps = [Computations(i, r, xs) for i in xs]
    n1s = [1.0, 1.5, 2.0, 2.5, 2.5, 1.5]
    n2s = [1.5, 2.0, 2.5, 2.5, 1.5, 1.0]
    for cmp, n1, n2 in zip(comps, n1s, n2s):
        assert cmp.n1 == n1
        assert cmp.n2 == n2


def test_total_internal_reflection():
    s = glass_sphere()
    r = Ray(point(0, 0, sqrt(2) / 2), vector(0, 1, 0))
    xs = [Intersection(-sqrt(2) / 2, s), Intersection(sqrt(2) / 2, s)]
    cmp = Computations(xs[1], r, xs)
    reflectance = schlick(cmp.eyev, cmp.normalv, cmp.n1, cmp.n2)
    assert abs(reflectance - 1.0) < EPSILON


def test_perpendicular_view_angle():
    s = glass_sphere()
    r = Ray(point(0, 0, 0), vector(0, 1, 0))
    xs = [Intersection(-1, s), Intersection(1, s)]
    cmp = Computations(xs[1], r, xs)
    reflectance = schlick(cmp.eyev, cmp.normalv, cmp.n1, cmp.n2)
    assert abs(reflectance - 0.04) < EPSILON


def test_small_view_angle():
    s = glass_sphere()
    r = Ray(point(0, 0.99, -2), vector(0, 0, 1))
    xs = [Intersection(1.8589, s)]
    cmp = Computations(xs[0], r, xs)
    reflectance = schlick(cmp.eyev, cmp.normalv, cmp.n1, cmp.n2)
    assert abs(reflectance - 0.48873) < EPSILON
