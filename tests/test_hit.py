from fancy_ray_tracer import Intersection, hit, Sphere


def test_hit_positive():
    s = Sphere()
    i1 = Intersection(1, s)
    i2 = Intersection(2, s)
    i = hit([i1, i2])
    assert i == i1


def test_hit_some_negative():
    s = Sphere()
    i1 = Intersection(-1, s)
    i2 = Intersection(2, s)
    i = hit([i1, i2])
    assert i == i2


def test_hit_all_negative():
    s = Sphere()
    i1 = Intersection(-2, s)
    i2 = Intersection(-1, s)
    i = hit([i1, i2])
    assert i is None


def test_hit():
    s = Sphere()
    i1 = Intersection(5, s)
    i2 = Intersection(7, s)
    i3 = Intersection(-3, s)
    i4 = Intersection(2, s)
    i = hit([i1, i2, i3, i4])
    assert i == i4
