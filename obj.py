from io import StringIO

from fancy_ray_tracer import *
from fancy_ray_tracer.constants import PI
from fancy_ray_tracer.parsers import WavefrontOBJ

with open('3d_objects/teapot_mild.obj') as f:
    data = StringIO(f.read())

wobj = WavefrontOBJ(data)
g = wobj.parse()
g.set_transform(chain_ops([rotX(4 * PI / 3), scaling(0.5, 0.5, 0.5)]))
box = make_box(g)


world = World(Light(point(-10, 10, -10), make_color(1, 1, 1)))
# world.add_object(g)
world.add_object(box)

hsize = 350
vsize = 200
camera = Camera(hsize, vsize, PI / 3)
camera.set_transform(view_transform(
    point(0, 2, -40), point(0, 1.5, 0), vector(0, 1, 0)))

canvas = Canvas((hsize, vsize))
camera.render(world, canvas)
# camera.render_sequential(world, canvas)
# canvas.save_img("scene.jpg")
canvas.canvas.show()
