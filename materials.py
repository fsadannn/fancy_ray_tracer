from fancy_ray_tracer import (
    Canvas,
    Light,
    Sphere,
    lighting,
    make_color,
    normalize,
    point,
    ray,
    scaling,
)
from fancy_ray_tracer.canvas import CanvasImg

canvas_pixels = 400
ray_origin = point(0, 0, -5)
wall_z = 10
wall_size = 7.0
pixel_size = wall_size / canvas_pixels
half = wall_size / 2

cv = CanvasImg('materials.jpg', (canvas_pixels, canvas_pixels))

color = make_color(1, 0, 0)
shape = Sphere()
shape.material.color = make_color(1, 0.2, 1)
#shape.set_transform(scaling(1, 0.5, 1))

light_position = point(-10, 10, -10)
light_color = make_color(1, 1, 1)
light = Light(light_position, light_color)

for y in range(canvas_pixels - 1):
    world_y = half - pixel_size * y

    for x in range(canvas_pixels - 1):
        world_x = -half + pixel_size * x

        position = point(world_x, world_y, wall_z)
        r = ray.Ray(ray_origin, normalize(position - ray_origin))
        xs = r.intersect(shape)
        hit = ray.hit(xs)
        if hit is not None:
            hit_point = r.position(hit.t)
            normal = ray.normal_at(hit.object, hit_point)
            eye = -r.direction
            color = lighting(hit.object.material,
                             light, hit_point, eye, normal)
            cv.set_pixelf(x, y, color.tolist())

cv.run()
