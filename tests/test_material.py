from math import sqrt

from fancy_ray_tracer import Light, lighting, make_material, point, vector
from fancy_ray_tracer.materials import StripePattern
from fancy_ray_tracer.primitives import Sphere
from fancy_ray_tracer.tuples import make_color
from fancy_ray_tracer.utils import ATOL, equal

MAT = make_material()
POS = point(0, 0, 0)
OBJ = Sphere()
OBJ.material = MAT


def test_light_eye_surface():
    eyev = vector(0, 0, -1)
    normalv = vector(0, 0, -1)
    light = Light(point(0, 0, -10), make_color(1, 1, 1))
    result = lighting(OBJ, light, POS, eyev, normalv)
    assert equal(result, make_color(1.9, 1.9, 1.9))


def test_light_eye45_surface():
    eyev = vector(0, sqrt(2) / 2, sqrt(2) / 2)
    normalv = vector(0, 0, -1)
    light = Light(point(0, 0, -10), make_color(1, 1, 1))
    result = lighting(OBJ, light, POS, eyev, normalv)
    assert equal(result, make_color(1.0, 1.0, 1.0))


def test_eye_light45_surface():
    eyev = vector(0, 0, -1)
    normalv = vector(0, 0, -1)
    light = Light(point(0, 10, -10), make_color(1, 1, 1))
    result = lighting(OBJ, light, POS, eyev, normalv)
    assert equal(result, make_color(0.7364, 0.7364, 0.7364))


def test_eyem45_light45_surface():
    eyev = vector(0, -sqrt(2) / 2, -sqrt(2) / 2)
    normalv = vector(0, 0, -1)
    light = Light(point(0, 10, -10), make_color(1, 1, 1))
    result = lighting(OBJ, light, POS, eyev, normalv)
    assert equal(result, make_color(1.6364, 1.6364, 1.6364))


def test_surface_shadow():
    eyev = vector(0, 0, -1)
    normalv = vector(0, 0, -1)
    light = Light(point(0, 0, -10), make_color(1, 1, 1))
    in_shadow = True
    result = lighting(OBJ, light, POS, eyev, normalv, in_shadow)
    assert equal(result, make_color(0.1, 0.1, 0.1))


def test_pattern():
    s = Sphere()
    m = make_material()
    m.pattern = StripePattern(make_color(1, 1, 1), make_color(0, 0, 0))
    m.ambient = 1
    m.diffuse = 0
    m.specular = 0
    s.material = m
    eyev = vector(0, 0, -1)
    normalv = vector(0, 0, -1)
    light = Light(point(0, 0, -10), make_color(1, 1, 1))
    in_shadow = False
    c1 = lighting(s, light, point(0.9, 0, 0), eyev, normalv, in_shadow)
    c2 = lighting(s, light, point(1.1, 0, 0), eyev, normalv, in_shadow)
    assert equal(c1, make_color(1, 1, 1))
    assert equal(c2, make_color(0, 0, 0))
