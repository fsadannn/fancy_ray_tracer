from math import sqrt
from typing import List

from fancy_ray_tracer import (
    Light,
    Ray,
    Sphere,
    World,
    equal,
    make_color,
    point,
    scaling,
    vector,
)
from fancy_ray_tracer.constants import ATOL
from fancy_ray_tracer.illumination import lighting
from fancy_ray_tracer.materials import DefaultPattern
from fancy_ray_tracer.matrices import translation
from fancy_ray_tracer.primitives import Plane
from fancy_ray_tracer.ray import Computations, Intersection, hit_sorted

SPHERE_CACHE = {}
DEFAULT_KEY = 'default'


def make_default_light():
    light_position = point(-10, 10, -10)
    light_color = make_color(1, 1, 1)
    light = Light(light_position, light_color)
    return light


def Spheres():
    s1 = Sphere()
    s1.material.color = make_color(0.8, 1, 0.6)
    s1.material.diffuse = 0.7
    s1.material.specular = 0.2
    s2 = Sphere()
    s2.set_transform(scaling(0.5, 0.5, 0.5))

    if DEFAULT_KEY in SPHERE_CACHE:
        id1, id2 = SPHERE_CACHE[DEFAULT_KEY]
        s1.id = id1
        s2.id = id2
    else:
        SPHERE_CACHE[DEFAULT_KEY] = (s1.id, s2.id)
    return (s1, s2)


def make_default_world():
    light = make_default_light()
    spheres = Spheres()
    w = World(light, spheres)
    return w


def test_world():
    light = make_default_light()
    w = make_default_world()
    s1, s2 = Spheres()
    assert w.light[0] == light
    assert w.has_object(s1)
    assert w.has_object(s2)


def test_world_intersection():
    w = make_default_world()
    r = Ray(point(0, 0, -5), vector(0, 0, 1))
    xs = w.intersec(r)
    assert len(xs) == 4
    assert abs(xs[0].t - 4) < ATOL
    assert abs(xs[1].t - 4.5) < ATOL
    assert abs(xs[2].t - 5.5) < ATOL
    assert abs(xs[3].t - 6) < ATOL


def test_world_shading():
    w = make_default_world()
    r = Ray(point(0, 0, -5), vector(0, 0, 1))
    shape = w.objects[0]
    i = Intersection(4, shape)
    cmp = Computations(i, r)
    c = lighting(cmp.object, w.light[0],
                 cmp.point, cmp.eyev, cmp.normalv, False)
    assert equal(c, make_color(0.38066, 0.47583, 0.2855))


def test_world_shading_inside():
    w = make_default_world()
    w.light = [Light(point(0, 0.25, 0), make_color(1, 1, 1))]
    r = Ray(point(0, 0, 0), vector(0, 0, 1))
    shape = w.objects[1]
    i = Intersection(0.5, shape)
    cmp = Computations(i, r)
    c = lighting(cmp.object, w.light[0],
                 cmp.point, cmp.eyev, cmp.normalv, False)
    assert equal(c, make_color(0.90498, 0.90498, 0.90498))


def test_world_color_ray_miss():
    w = make_default_world()
    r = Ray(point(0, 0, -5), vector(0, 1, 0))
    intersections = w.intersec(r)
    it = hit_sorted(intersections)
    if it is None:
        c = make_color(0, 0, 0)
    else:
        cmp = Computations(it, r)
        c = lighting(cmp.object, w.light[0],
                     cmp.point, cmp.eyev, cmp.normalv, False)

    assert equal(c, make_color(0, 0, 0))


def test_world_color_ray_hits():
    w = make_default_world()
    r = Ray(point(0, 0, -5), vector(0, 0, 1))
    intersections = w.intersec(r)
    it = hit_sorted(intersections)
    if it is None:
        c = make_color(0, 0, 0)
    else:
        cmp = Computations(it, r)
        c = lighting(cmp.object, w.light[0],
                     cmp.point, cmp.eyev, cmp.normalv, False)
    assert equal(c, make_color(0.38066, 0.47583, 0.2855))


def test_world_color_ray_hits_behind():
    w = make_default_world()
    outer = w.objects[0]
    outer.material.ambient = 1.0
    inner = w.objects[1]
    inner.material.ambient = 1.0
    r = Ray(point(0, 0, 0.75), vector(0, 0, -1))
    c = w.color_at(r)
    assert equal(c, inner.material.color)


def test_shadowed_no_colineal():
    w = make_default_world()
    p = point(0, 10, 0)
    assert w.is_shadowed(p) == False


def test_shadowed_object_between():
    w = make_default_world()
    p = point(10, -10, 10)
    assert w.is_shadowed(p) == True


def test_shadowed_object_behind_light():
    w = make_default_world()
    p = point(-20, 20, -20)
    assert w.is_shadowed(p) == False


def test_shadowed_object_behind_point():
    w = make_default_world()
    p = point(-2, 2, -2)
    assert w.is_shadowed(p) == False


def test_shade_hit():
    w = make_default_world()
    w.light = [Light(point(0, 0, -10), make_color(1, 1, 1))]
    s1 = Sphere()
    w.add_object(s1)
    s2 = Sphere()
    s2.set_transform(translation(0, 0, 10))
    w.add_object(s2)
    r = Ray(point(0, 0, 5), vector(0, 0, 1))
    i = Intersection(4, s2)
    cmp = Computations(i, r)
    c = w.shade_hit(cmp)
    assert equal(c, make_color(0.1, 0.1, 0.1))


def test_reflection_nonreflective():
    w = make_default_world()
    r = Ray(point(0, 0, 0), vector(0, 0, 1))
    s = w.objects[1]
    s.material.ambient = 1
    i = Intersection(1, s)
    cmp = Computations(i, r)
    assert equal(w.reflected_color(cmp), make_color(0, 0, 0))


def test_non_infinity_recursion():
    w = make_default_world()
    w.light = [Light(point(0, 0, 0), make_color(1, 1, 1))]
    lower = Plane()
    lower.material.reflective = 1
    lower.set_transform(translation(0, -1, 0))
    w.add_object(lower)
    upper = Plane()
    upper.material.reflective = 1
    upper.set_transform(translation(0, 1, 0))
    w.add_object(upper)
    r = Ray(point(0, 0, 0), vector(0, 0, 1))
    c = w.color_at(r)


def test_refraction_opaque():
    w = make_default_world()
    r = Ray(point(0, 0, -5), vector(0, 0, 1))
    s = w.objects[0]
    xs = [Intersection(4, s), Intersection(6, s)]
    cmp = Computations(xs[0], r, xs)
    assert equal(w.refracted_color(cmp), make_color(0, 0, 0))


def test_refraction_at_max_depth():
    w = make_default_world()
    r = Ray(point(0, 0, -5), vector(0, 0, 1))
    s = w.objects[0]
    s.material.transparency = 1.0
    s.material.refractive_index = 1.5
    xs = [Intersection(4, s), Intersection(6, s)]
    cmp = Computations(xs[0], r, xs)
    assert equal(w.refracted_color(cmp, 0), make_color(0, 0, 0))


def test_total_internal_refraction():
    w = make_default_world()
    s = w.objects[0]
    s.material.transparency = 1.0
    s.material.refractive_index = 1.5
    r = Ray(point(0, 0, sqrt(2) / 2), vector(0, 1, 0))
    xs = [Intersection(-sqrt(2) / 2, s), Intersection(sqrt(2) / 2, s)]
    cmp = Computations(xs[1], r, xs)
    assert equal(w.refracted_color(cmp), make_color(0, 0, 0))


def test_refraction():
    w = make_default_world()
    A = w.objects[0]
    A.material.ambient = 1.0
    A.material.pattern = DefaultPattern()
    B = w.objects[1]
    B.material.transparency = 1.0
    B.material.refractive_index = 1.5
    r = Ray(point(0, 0, 0.1), vector(0, 1, 0))
    xs = [Intersection(-0.9899, A), Intersection(-0.4899, B),
          Intersection(0.4899, B), Intersection(0.9899, A)]
    cmp = Computations(xs[2], r, xs)
    c = w.refracted_color(cmp)
    assert equal(c, make_color(0, 0.99888, 0.04725), 1e-4, 1e-4)


def test_refraction2():
    w = make_default_world()
    floor = Plane()
    floor.set_transform(translation(0, -1, 0))
    floor.material.transparency = 0.5
    floor.material.refractive_index = 1.5
    w.add_object(floor)
    ball = Sphere()
    ball.material.color = make_color(1, 0, 0)
    ball.material.ambient = 0.5
    ball.set_transform(translation(0, -3.5, -0.5))
    w.add_object(ball)
    r = Ray(point(0, 0, -3), vector(0, -sqrt(2) / 2, sqrt(2) / 2))
    xs = [Intersection(sqrt(2), floor)]
    cmp = Computations(xs[0], r, xs)
    color = w.shade_hit(cmp)
    assert equal(color, make_color(0.93642, 0.68642, 0.68642))


def test_refraction_reflection():
    w = make_default_world()
    r = Ray(point(0, 0, -3), vector(0, -sqrt(2) / 2, sqrt(2) / 2))
    floor = Plane()
    floor.set_transform(translation(0, -1, 0))
    floor.material.reflective = 0.5
    floor.material.transparency = 0.5
    floor.material.refractive_index = 1.5
    w.add_object(floor)
    ball = Sphere()
    ball.material.color = make_color(1, 0, 0)
    ball.material.ambient = 0.5
    ball.set_transform(translation(0, -3.5, -0.5))
    w.add_object(ball)
    xs = [Intersection(sqrt(2), floor)]
    cmp = Computations(xs[0], r, xs)
    color = w.shade_hit(cmp)
    assert equal(color, make_color(0.93391, 0.69643, 0.69243))
