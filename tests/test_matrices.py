from math import pi, sqrt

from fancy_ray_tracer import matrices, tuples


def test_translation():
    transform = matrices.translation(5, -3, 2)
    inv = matrices.inverse(transform)
    p = tuples.point(-3, 4, 5)
    v = tuples.vector(-3, 4, 5)
    assert tuples.equal(transform.dot(
        p), tuples.point(2, 1, 7))
    assert tuples.equal(inv.dot(
        p), tuples.point(-8, 7, 3))
    assert tuples.equal(transform.dot(
        v), v)


def test_scaling():
    transform = matrices.scaling(2, 3, 4)
    inv = matrices.inverse(transform)
    p = tuples.point(-4, 6, 8)
    v = tuples.vector(-4, 6, 8)
    assert tuples.equal(transform.dot(
        p), tuples.point(-8, 18, 32))
    assert tuples.equal(inv.dot(
        p), tuples.point(-2, 2, 2))
    assert tuples.equal(transform.dot(
        v), tuples.vector(-8, 18, 32))


def test_rotX():
    p = tuples.point(0, 1, 0)
    half_quarter = matrices.rotX(pi / 4)
    full_quarter = matrices.rotX(pi / 2)
    assert tuples.equal(half_quarter.dot(
        p), tuples.point(0, sqrt(2) / 2, sqrt(2) / 2))
    assert tuples.equal(full_quarter.dot(
        p), tuples.point(0, 0, 1))


def test_rotX_inv():
    p = tuples.point(0, 1, 0)
    half_quarter = matrices.rotX(pi / 4)
    half_quarter_inv = matrices.inverse(half_quarter)
    assert tuples.equal(half_quarter_inv.dot(
        p), tuples.point(0, (sqrt(2) / 2), -(sqrt(2) / 2)))


def test_rotY():
    p = tuples.point(0, 0, 1)
    half_quarter = matrices.rotY(pi / 4)
    full_quarter = matrices.rotY(pi / 2)
    assert tuples.equal(half_quarter.dot(
        p), tuples.point(sqrt(2) / 2, 0, sqrt(2) / 2))
    assert tuples.equal(full_quarter.dot(
        p), tuples.point(1, 0, 0))


def test_rotZ():
    p = tuples.point(0, 1, 0)
    half_quarter = matrices.rotZ(pi / 4)
    full_quarter = matrices.rotZ(pi / 2)
    assert tuples.equal(half_quarter.dot(
        p), tuples.point(-(sqrt(2) / 2), sqrt(2) / 2, 0))
    assert tuples.equal(full_quarter.dot(
        p), tuples.point(-1, 0, 0))


def test_sharing():
    transform = matrices.sharing(1, 0, 0, 0, 0, 0)
    p = tuples.point(2, 3, 4)
    assert tuples.equal(transform.dot(
        p), tuples.point(5, 3, 4))

    transform = matrices.sharing(0, 1, 0, 0, 0, 0)
    assert tuples.equal(transform.dot(
        p), tuples.point(6, 3, 4))

    transform = matrices.sharing(0, 0, 1, 0, 0, 0)
    assert tuples.equal(transform.dot(
        p), tuples.point(2, 5, 4))

    transform = matrices.sharing(0, 0, 0, 1, 0, 0)
    assert tuples.equal(transform.dot(
        p), tuples.point(2, 7, 4))

    transform = matrices.sharing(0, 0, 0, 0, 1, 0)
    assert tuples.equal(transform.dot(
        p), tuples.point(2, 3, 6))

    transform = matrices.sharing(0, 0, 0, 0, 0, 1)
    assert tuples.equal(transform.dot(
        p), tuples.point(2, 3, 7))
