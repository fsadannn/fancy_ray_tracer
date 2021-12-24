from math import sqrt

from fancy_ray_tracer import (
    Light,
    Material,
    intersect,
    lighting,
    make_material,
    make_sphere,
    point,
    scaling,
    vector,
)
from fancy_ray_tracer.tuples import ATOL, color, equal

MAT = make_material()
POS = point(0, 0, 0)


def test_light_eye_surface():
    eyev = vector(0, 0, -1)
    normalv = vector(0, 0, -1)
    light = Light(point(0, 0, -10), color(1, 1, 1))
    result = lighting(MAT, light, POS, eyev, normalv)
    assert equal(result, color(1.9, 1.9, 1.9))


def test_light_eye45_surface():
    eyev = vector(0, sqrt(2) / 2, sqrt(2) / 2)
    normalv = vector(0, 0, -1)
    light = Light(point(0, 0, -10), color(1, 1, 1))
    result = lighting(MAT, light, POS, eyev, normalv)
    assert equal(result, color(1.0, 1.0, 1.0))


def test_eye_light45_surface():
    eyev = vector(0, 0, -1)
    normalv = vector(0, 0, -1)
    light = Light(point(0, 10, -10), color(1, 1, 1))
    result = lighting(MAT, light, POS, eyev, normalv)
    assert equal(result, color(0.7364, 0.7364, 0.7364))


def test_eyem45_light45_surface():
    eyev = vector(0, -sqrt(2) / 2, -sqrt(2) / 2)
    normalv = vector(0, 0, -1)
    light = Light(point(0, 10, -10), color(1, 1, 1))
    result = lighting(MAT, light, POS, eyev, normalv)
    assert equal(result, color(1.6364, 1.6364, 1.6364))
