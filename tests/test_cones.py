# from fancy_ray_tracer import *
# from fancy_ray_tracer.constants import EPSILON


# def test_intersect():
#     c = Cylinder()

#     origins = [
#         point(0, 0, -5),
#         point(0, 0, -5),
#         point(1, 1, -5),
#     ]
#     directions = [
#         vector(0, 0, 1),
#         vector(1, 1, 1),
#         vector(-0.5, -1, 1),
#     ]
#     t1s = [5, 4, 6.80798]
#     t2s = [5, 6, 7.08872]
#     for origin, direction, t1, t2 in zip(origins, directions, t1s, t2s):
#         its = c.intersect(origin, normalize(direction))
#         assert len(its) == 2
#         assert abs(its[0].t - t1) < EPSILON
#         assert abs(its[1].t - t2) < EPSILON

# def test_miss():
#     c = Cylinder()

#     origins = [
#         point(1, 0, 0),
#         point(0, 0, 0),
#         point(0, 0, -5),
#     ]
#     directions = [
#         vector(0, 1, 0),
#         vector(0, 1, 0),
#         vector(1, 1, 1),
#     ]
#     for origin, direction in zip(origins, directions):
#         its = c.intersect(origin, normalize(direction))
#         assert len(its) == 0


# def test_truncated():
#     c = Cylinder(1, 2)

#     origins = [
#         point(0, 1.5, 0),
#         point(0, 3, -5),
#         point(0, 0, -5),
#         point(0, 2, -5),
#         point(0, 1, -5),
#         point(0, 1.5, -1),
#     ]
#     directions = [
#         vector(0.1, 1, 0),
#         vector(0, 0, 1),
#         vector(0, 0, 1),
#         vector(0, 0, 1),
#         vector(0, 0, 1),
#         vector(0, 0, 1),
#     ]
#     counts = [0, 0, 0, 0, 0, 2]
#     for origin, direction, count in zip(origins, directions, counts):
#         its = c.intersect(origin, normalize(direction))
#         assert len(its) == count


# def test_closed():
#     c = Cylinder(1, 2, True)

#     origins = [
#         point(0, 3, 0),
#         point(0, 3, -2),
#         point(0, 4, -2),
#         point(0, 0, -2),
#         point(0, -1, -2),
#     ]
#     directions = [
#         vector(0, -1, 0),
#         vector(0, -1, 2),
#         vector(0, -1, 1),
#         vector(0, 1, 2),
#         vector(0, 1, 1),
#     ]
#     for origin, direction in zip(origins, directions):
#         its = c.intersect(origin, normalize(direction))
#         print(origin, direction)
#         assert len(its) == 2
