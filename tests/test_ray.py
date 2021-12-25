from math import sqrt

from fancy_ray_tracer import (
    Ray,
    equal,
    make_sphere,
    point,
    scaling,
    transform,
    translation,
    utils,
    vector,
)
from fancy_ray_tracer.matrices import identity, rotZ
from fancy_ray_tracer.ray import normal_at, reflect
from fancy_ray_tracer.tuples import normalize


def test_ray():
    p = point(2, 3, 4)
    ray = Ray(p, vector(1, 0, 0))
    assert equal(ray.position(0), p)
    assert equal(ray.position(1), point(3, 3, 4))
    assert equal(ray.position(-1), point(1, 3, 4))
    assert equal(ray.position(2.5), point(4.5, 3, 4))


def test_transform():
    ray = Ray(point(1, 2, 3), vector(0, 1, 0))
    m = translation(3, 4, 5)
    r2 = transform(ray, m)
    assert equal(r2.origin, point(4, 6, 8))
    assert equal(r2.direction, vector(0, 1, 0))

    m = scaling(2, 3, 4)
    r2 = transform(ray, m)
    assert equal(r2.origin, point(2, 6, 12))
    assert equal(r2.direction, vector(0, 3, 0))


def test_sphere_default_transform():
    s = make_sphere()
    assert equal(s.transform, identity())


def test_sphere_change_transform():
    s = make_sphere()
    t = translation(2, 3, 4)
    s.set_transform(t)
    assert equal(s.transform, t)


def test_normal_sphere_x():
    s = make_sphere()
    n = normal_at(s, point(1, 0, 0))
    assert equal(n, vector(1, 0, 0))


def test_normal_sphere_y():
    s = make_sphere()
    n = normal_at(s, point(0, 1, 0))
    assert equal(n, vector(0, 1, 0))


def test_normal_sphere_z():
    s = make_sphere()
    n = normal_at(s, point(0, 0, 1))
    assert equal(n, vector(0, 0, 1))


def test_normal_sphere_nonaxial():
    s = make_sphere()
    val = sqrt(3) / 3
    n = normal_at(s, point(val, val, val))
    assert equal(n, vector(val, val, val))


def test_normal_sphere_normalize():
    s = make_sphere()
    val = sqrt(3) / 3
    n = normal_at(s, point(val, val, val))
    assert equal(n, normalize(n))


def test_normal_translated_sphere():
    s = make_sphere()
    s.set_transform(translation(0, 1, 0))
    n = normal_at(s, point(0, 1.70711, -0.70711))
    assert equal(n, vector(0, 0.70711, -0.70711))


def test_normal_transformed_sphere():
    s = make_sphere()
    m = scaling(1, 0.5, 1).dot(rotZ(utils.pi / 5))
    s.set_transform(m)
    n = normal_at(s, point(0, sqrt(2) / 2, -sqrt(2) / 2))
    assert equal(n, vector(0, 0.97014, -0.24254))


def test_reflect_45deg():
    v = vector(1, -1, 0)
    n = vector(0, 1, 0)
    r = reflect(v, n)
    assert equal(r, vector(1, 1, 0))


def test_reflect_slanted_surface():
    v = vector(0, -1, 0)
    n = vector(sqrt(2) / 2, sqrt(2) / 2, 0)
    r = reflect(v, n)
    assert equal(r, vector(1, 0, 0))