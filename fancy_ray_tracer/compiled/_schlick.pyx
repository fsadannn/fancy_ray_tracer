from cython cimport boundscheck, wraparound
from libc cimport math

@boundscheck(False)
@wraparound(False)
cpdef double schlick(double[:] eyev,double[:] normalv,double n1,double n2) :  # reflectance
    cos: double = eyev[0]*normalv[0]+eyev[1]*normalv[1]+eyev[2]*normalv[2]

    if n1 > n2:
        sin2_t = math.pow((n1 / n2),2.0) * (1.0 - math.pow(cos,2.0))

        if sin2_t > 1.0:
            return 1.0

        cos = math.sqrt(1.0 - sin2_t)

    r0 = math.pow(((n1 - n2) / (n1 + n2)),2.0)
    return r0 + (1 - r0) * math.pow((1 - cos),5.0)