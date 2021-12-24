from math import sqrt

from fancy_ray_tracer import Canvas, CanvasImg, matrices, tuples, utils

cv = CanvasImg('clock.jpg')

orig = tuples.point(0, 1, 0)
radius: float = 3 / 4 * (min(cv.width, cv.height) / 2)
scale = matrices.scaling(radius, radius, 0)
translate = matrices.translation(cv.width / 2, cv.height / 2, 0)
point = matrices.rotZ(utils.pi).dot(translate.dot(orig))
for i in range(12):
    point = utils.chain(
        [matrices.rotZ(utils.pi * (i / 6)), scale, translate], orig)
    cv.set_pixel(int(point[0]), int(point[1]), (255, 255, 255))

cv.run()
