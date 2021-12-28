from cython cimport boundscheck, wraparound, cdivision
from libc cimport math

@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef double schlick(double[:] eyev,double[:] normalv,double n1,double n2) :  # reflectance
    cdef:
        double cos = eyev[0]*normalv[0]+eyev[1]*normalv[1]+eyev[2]*normalv[2]
        double temp
        double sin2_t
        double r0

    if n1 > n2:
        temp = n1 / n2
        sin2_t = temp*temp * (1.0 - cos*cos)

        if sin2_t > 1.0:
            return 1.0

        cos = math.sqrt(1.0 - sin2_t)

    r0 = ((n1 - n2) / (n1 + n2))
    r0 *= r0
    return r0 + (1 - r0) * math.pow((1 - cos),5.0)
