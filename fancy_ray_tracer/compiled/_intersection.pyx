from cython cimport boundscheck, wraparound, cdivision
from libc cimport math

@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef tuple check_axis(double origin, double direction, double epsilon, double axis_min = -1.0, double axis_max = 1.0):
    cdef double tmin
    cdef double tmax
    cdef double temp

    if math.fabs(direction)>=epsilon:
        tmin = (axis_min-origin)/direction
        tmax = (axis_max-origin)/direction
    else:
        tmin = (axis_min-origin)*math.INFINITY
        tmax = (axis_max-origin)*math.INFINITY

    if tmin > tmax:
        temp = tmin
        tmin = tmax
        tmax = temp

    return (tmin,tmax)
