from math import sqrt

from fancy_ray_tracer import Camera, equal, point, vector
from fancy_ray_tracer.constants import PI
from fancy_ray_tracer.matrices import rotY, translation
from fancy_ray_tracer.utils import chain, chain_ops


def test_h_canvas():
    c = Camera(200, 125, PI / 2)
    assert abs(c.pixel_size - 0.01)


def test_v_canvas():
    c = Camera(125, 200, PI / 2)
    assert abs(c.pixel_size - 0.01)


def test_ray_center():
    c = Camera(201, 101, PI / 2)
    r = c.ray_for_pixel(100, 50)
    assert equal(r.origin, point(0, 0, 0))
    assert equal(r.direction, vector(0, 0, -1))


def test_ray_center2():
    c = Camera(201, 101, PI / 2)
    r = c.ray_for_pixel(0, 0)
    assert equal(r.origin, point(0, 0, 0))
    assert equal(r.direction, vector(0.66519, 0.33259, -0.66851))


def test_ray_moved_camera():
    c = Camera(201, 101, PI / 2)
    c.set_transform(chain_ops([rotY(PI / 4), translation(0, -2, 5)]))
    r = c.ray_for_pixel(100, 50)
    assert equal(r.origin, point(0, 2, -5))
    assert equal(r.direction, vector(sqrt(2) / 2, 0, -sqrt(2) / 2))
