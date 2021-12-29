from cython cimport boundscheck, wraparound, cdivision
from libc cimport math

@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef tuple aabb_box_intersect(double[:] bound_min, double[:] bound_max,
        double[:] origin, double[:] direction, double epsilon): # smith method
    cdef:
        double tmin
        double tmax
        double tminy
        double tmaxy
        double tminz
        double tmaxz
        double temp

    temp = direction[0]
    if math.fabs(temp)>=epsilon:
        temp = 1/temp
    else:
        temp = math.INFINITY

    if temp>=0:
        tmin = (bound_min[0]-origin[0])*temp
        tmax = (bound_max[0]-origin[0])*temp
    else:
        tmin = (bound_max[0]-origin[0])*temp
        tmax = (bound_min[0]-origin[0])*temp

    temp = direction[1]
    if math.fabs(temp)>=epsilon:
        temp = 1/temp;
    else:
        temp = math.INFINITY

    if temp>=0:
        tminy = (bound_min[1]-origin[1])*temp
        tmaxy = (bound_max[1]-origin[1])*temp
    else:
        tminy = (bound_max[1]-origin[1])*temp
        tmaxy = (bound_min[1]-origin[1])*temp

    if tmin>tmaxy or tminy>tmax:
        return None
    if tminy > tmin:
        tmin = tminy
    if tmaxy < tmax:
        tmax = tmaxy

    temp = direction[2]
    if math.fabs(temp)>=epsilon:
        temp = 1/temp;
    else:
        temp = math.INFINITY

    if temp>=0:
        tminz = (bound_min[2]-origin[2])*temp
        tmaxz = (bound_max[2]-origin[2])*temp
    else:
        tminz = (bound_max[2]-origin[2])*temp
        tmaxz = (bound_min[2]-origin[2])*temp

    if tmin>tmaxz or tminz>tmax:
        return None
    if tminz > tmin:
        tmin = tminz
    if tmaxz < tmax:
        tmax = tmaxz

    return (tmin,tmax)
