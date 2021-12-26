from fancy_ray_tracer import *
from fancy_ray_tracer.constants import PI

floor = Plane()
floor.material = make_material()
floor.material.color = make_color(1, 0.9, 0.9)
floor.material.specular = 0.0
#floor.material.reflective = 0.8
floor.material.pattern = ChessPattern(
    make_color(0, 0, 0), make_color(1, 1, 1))


left_wall = Plane()
left_wall.set_transform(chain_ops(
    [translation(0, 0, 5), rotY(-PI / 4), rotX(PI / 2)]))
left_wall.material = make_material()
left_wall.material.color = make_color(1, 0.9, 0.9)
left_wall.material.specular = 0.0
left_wall.material.pattern = ChessPattern(
    make_color(0, 0, 0), make_color(1, 1, 1))

right_wall = Plane()
right_wall.set_transform(chain_ops(
    [translation(0, 0, 5), rotY(PI / 4), rotX(PI / 2)]))
right_wall.material = left_wall.material


middle = Sphere()
middle.set_transform(translation(-0.5, 1, 0.5))
middle.material = make_material()
middle.material.color = make_color(0.1, 1, 0.5)
middle.material.diffuse = 0.7
middle.material.specular = 0.3
middle.material.reflective = 0.8

right = Sphere()
right.set_transform(
    chain_ops([translation(1.5, 0.5, -0.5), scaling(0.5, 0.5, 0.5)]))
right.material = make_material()
right.material.color = make_color(0.5, 1, 0.1)
right.material.diffuse = 0.7
right.material.specular = 0.3


left = Sphere()
left.set_transform(
    chain_ops([translation(-1.5, 0.33, -0.75), scaling(0.33, 0.33, 0.33), ]))
left.material = make_material()
left.material.color = make_color(1, 0.8, 0.1)
left.material.diffuse = 0.7
left.material.specular = 0.3

world = World(Light(point(-10, 10, -10), make_color(1, 1, 1)))
world.add_object(floor)
world.add_object(left_wall)
world.add_object(right_wall)
world.add_object(middle)
world.add_object(right)
world.add_object(left)

hsize = 300
vsize = 200
camera = Camera(hsize, vsize, PI / 3)
camera.set_transform(view_transform(
    point(0, 1.5, -5), point(0, 1, 0), vector(0, 1, 0)))

canvas = Canvas((hsize, vsize))
#canvas = CanvasImg('scene.jpg', (hsize, vsize))

camera.render(world, canvas)

canvas.run()
