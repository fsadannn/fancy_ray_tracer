from fancy_ray_tracer import Light, Sphere, make_color, point, scaling


def make_default_wrold():
    light_position = point(-10, 10, -10)
    light_color = make_color(1, 1, 1)
    light = Light(light_position, light_color)
    s1 = Sphere()
    s1.material.color = make_color(0.8, 1, 0.6)
    s1.material.diffuse = 0.7
    s1.material.specular = 0.2
    s2 = Sphere()
    s2.set_transform(scaling(0.5, 0.5, 0.5))
