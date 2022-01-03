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


middle = Cube()
middle.set_transform(
    chain_ops([translation(0, 1.8, -0.5), rotX(PI / 4), rotZ(PI / 4)]))
middle.material = make_material()
middle.material.color = make_color(0.8, 0, 0)
middle.material.diffuse = 0.7
middle.material.specular = 0.3
middle.material.reflective = 0.8
# middle.material.refractive_index = 1.5
# middle.material.transparency = 1.0
# middle_box = make_box(middle)

right = Cylinder(0, 2, closed=True)
right.set_transform(
    chain_ops([translation(1.5, 1, -1.8), scaling(0.6, 0.5, 0.6), rotX(-PI / 8)]))
right.material = make_material()
right.material.color = make_color(0.5, 1, 0.1)
right.material.diffuse = 0.7
right.material.specular = 0.3
right.material.reflective = 0.2
#right.material.refractive_index = 1.5
#right.material.transparency = 0.9
# right_box = make_box(right)
#right.has_shadow = False

left = Sphere()
left.set_transform(
    chain_ops([translation(-1.5, 0.33, -0.75), scaling(0.33, 0.33, 0.33), ]))
left.material = make_material()
left.material.color = make_color(1, 0.8, 0.1)
left.material.diffuse = 0.7
left.material.specular = 0.3
# left_box = make_box(left)

other = Cone(-0.5, 0.5, closed=True)
other.set_transform(
    chain_ops([translation(-2, 1.5, -0.75), rotX(PI / 6), rotZ(PI / 6)]))
other.material = make_material()
other.material.color = make_color(0, 0, 0.8)
other.material.diffuse = 0.7
other.material.specular = 0.3
other.material.reflective = 0.6
other_box = make_box(other)
# print(other_box.bound_min, other_box.bound_max)

g = Group()
g.add_shape(middle)
g.add_shape(left)
g.add_shape(right)
g.add_shape(other_box)
box = make_box(g)

pp = Plane()
pp.set_transform(chain_ops([translation(0, 0, -4), rotX(-PI / 2)]))
pp.material.reflective = 0.9
pp.material.transparency = 1.0
pp.material.refractive_index = 1.5
pp.material.ambient = 0.1
pp.material.diffuse = 0.1
pp.material.color = make_color(0.2, 0, 0.2)
pp.has_shadow = False

world = World(Light(point(-10, 10, -10), make_color(1, 1, 1)))
world.add_object(floor)
world.add_object(left_wall)
world.add_object(right_wall)
world.add_object(g)
# world.add_object(pp)
# world.add_object(middle)
# world.add_object(right)
# world.add_object(left)
# world.add_object(other)
# world.add_object(middle_box)
# world.add_object(right_box)
# world.add_object(left_box)
# world.add_object(other_box)

hsize = 350
vsize = 225
camera = Camera(hsize, vsize, PI / 3)
camera.set_transform(view_transform(
    point(0, 2, -7), point(0, 1.5, 0), vector(0, 1, 0)))

canvas = Canvas((hsize, vsize))
camera.render(world, canvas)
# camera.render_sequential(world, canvas)
# canvas.save_img("scene.jpg")
canvas.canvas.show()
print(camera.pixel_size)
