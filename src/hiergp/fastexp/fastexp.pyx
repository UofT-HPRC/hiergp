import numpy as np
cimport numpy as np
from libc.math cimport exp

cdef extern from "fastexp.h":
    void ffastexp(double *values, int N) nogil

def fastexp(np.ndarray[dtype=np.float64_t] a):
    """In-place fast exponential of floating point numbers."""
    if not (a.flags['C_CONTIGUOUS'] or
            a.flags['F_CONTIGUOUS']):
        raise TypeError('Not Contiguous')


    ffastexp(&a[0], a.size)
