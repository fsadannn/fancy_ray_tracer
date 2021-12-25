from fancy_ray_tracer import (
    Light,
    Ray,
    Sphere,
    World,
    make_color,
    point,
    scaling,
    vector,
)
from fancy_ray_tracer.constants import ATOL

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
    assert w.light == light
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
