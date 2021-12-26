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
from fancy_ray_tracer.matrices import translation
from fancy_ray_tracer.ray import Computations, Intersection, hit_sorted

SPHERE_CACHE = {}
DEFAULT_KEY = 'default'


def make_default_light():
    light_position = point(-10, 10, -10)
    light_color = make_color(1, 1, 1)
    light = Light(light_position, light_color)
    return light


def make_spheres():
    if DEFAULT_KEY in SPHERE_CACHE:
        return SPHERE_CACHE[DEFAULT_KEY]

    s1 = Sphere()
    s1.material.color = make_color(0.8, 1, 0.6)
    s1.material.diffuse = 0.7
    s1.material.specular = 0.2
    s2 = Sphere()
    s2.set_transform(scaling(0.5, 0.5, 0.5))

    SPHERE_CACHE[DEFAULT_KEY] = [s1, s2]

    return [s1, s2]


def make_default_world():
    light = make_default_light()
    spheres = make_spheres()
    w = World(light, spheres)
    return w


def test_world():
    light = make_default_light()
    w = make_default_world()
    s1, s2 = make_spheres()
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
    c = lighting(cmp.object.material, w.light[0],
                 cmp.point, cmp.eyev, cmp.normalv, False)
    assert equal(c, make_color(0.38066, 0.47583, 0.2855))


def test_world_shading_inside():
    w = make_default_world()
    w.light = [Light(point(0, 0.25, 0), make_color(1, 1, 1))]
    r = Ray(point(0, 0, 0), vector(0, 0, 1))
    shape = w.objects[1]
    i = Intersection(0.5, shape)
    cmp = Computations(i, r)
    c = lighting(cmp.object.material, w.light[0],
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
        c = lighting(cmp.object.material, w.light[0],
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
        c = lighting(cmp.object.material, w.light[0],
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
