from fancy_ray_tracer import Canvas

cv = Canvas()

with cv.get_pixel_array_cm() as pxs:
    pxs[20:100, 20:100] = (250, 0, 50)


def udx(cv: Canvas):
    cv.to_img('test.jpg')
    cv.stop()


cv.set_update(udx)
cv.run()
