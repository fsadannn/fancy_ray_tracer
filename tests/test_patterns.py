from fancy_ray_tracer import *
from fancy_ray_tracer.materials import LinearGradient, StripePattern

BLACK = make_color(0, 0, 0)
WHITE = make_color(1, 1, 1)


def stripes_obj_transform():
    obj = Sphere()
    obj.set_transform(scaling(2, 2, 2))
    p = StripePattern(WHITE, BLACK)
    c = obj.color_at(point(1.5, 0, 0))
    assert equal(c, WHITE)


def stripes_pattern_transform():
    obj = Sphere()
    p = StripePattern(WHITE, BLACK)
    p.set_transform(scaling(2, 2, 2))
    c = obj.color_at(point(1.5, 0, 0))
    assert equal(c, WHITE)


def stripes_pattern_obj_transform():
    obj = Sphere()
    obj.set_transform(scaling(2, 2, 2))
    p = StripePattern(WHITE, BLACK)
    p.set_transform(translation(0.5, 0, 0))
    c = obj.color_at(point(2.5, 0, 0))
    assert equal(c, WHITE)


def linear_gradient():
    p = LinearGradient(WHITE, BLACK)
    assert equal(p.color_at(point(0, 0, 0)), WHITE)
    assert equal(p.color_at(point(0.25, 0, 0)), make_color(0.75, 0.75, 0.75))
    assert equal(p.color_at(point(0.5, 0, 0)), make_color(0.5, 0.5, 0.5))
    assert equal(p.color_at(point(0.75, 0, 0)), make_color(0.25, 0.25, 0.25))
