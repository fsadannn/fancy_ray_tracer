from fancy_ray_tracer import *
from fancy_ray_tracer.constants import PI


def hexagon_corner():
    corner = Sphere()
    corner.set_transform(
        chain_ops([translation(0, 0, -1), scaling(0.25, 0.25, 0.25)]))
    return corner


def hexagon_edge():
    edge = Cylinder(0, 1)
    edge.set_transform(chain_ops(
        [translation(0, 0, -1), rotY(-PI / 6), rotZ(-PI / 2), scaling(0.25, 1, 0.25)]))
    return edge


def hexagon_side():
    side = Group()
    side.add_shape(hexagon_corner())
    side.add_shape(hexagon_edge())
    return make_box(side)


def hexagon():
    hex = Group()

    for i in range(6):
        side = hexagon_side()
        side.set_transform(rotY(i * PI / 3))
        hex.add_shape(side)

    return make_box(hex)


hex = hexagon()

world = World(Light(point(-10, 10, -10), make_color(1, 1, 1)))
world.add_object(hex)

hsize = 350
vsize = 225
camera = Camera(hsize, vsize, PI / 3)
camera.set_transform(view_transform(
    point(0, 3, -5), point(0, 0, 0), vector(0, 1, 0)))

canvas = Canvas((hsize, vsize))
camera.render(world, canvas)

canvas.canvas.show()
