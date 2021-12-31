from fancy_ray_tracer import *


def test_bound():
    g = Group()
    s = Sphere()
    c = Cube()
    c.set_transform(translation(1, 0, 0))
    g.add_shape(s)
    g.add_shape(c)
    b = make_box(g)
    assert equal(b.bound_min, point(-1, -1, -1))
    assert equal(b.bound_max, point(2, 1, 1))
