# Cython Pairs function
# ------------------------

from __future__ import division
import numpy as np
# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np
cimport cython
np.import_array()
# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.double
LONG = np.long
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.double_t DTYPE_t

ctypedef np.long_t LONG_t
# "def" can type its arguments but not have a return type. The type of the
# arguments for a "def" function is checked at run-time when entering the
# function.

def count_frequency(np.ndarray[LONG_t, ndim=2] X, unsigned int d):
    cdef unsigned int N = X.shape[0]
    cdef unsigned int W = X.shape[1]

    cdef np.ndarray[LONG_t, ndim=2] Y = np.zeros( (N,d), dtype=LONG )

    for n in range( N ):
        for w in range( W ):
            i = X[n,w]
            Y[n,i] += 1

    return Y

@cython.boundscheck(False) 
def Pairs(np.ndarray[DTYPE_t, ndim=2] x1, np.ndarray[DTYPE_t, ndim=2] x2):
    """Compute E[x1 \ctimes x2]"""

    assert x1.dtype == DTYPE and x2.dtype == DTYPE

    cdef unsigned int N = x1.shape[0]
    cdef unsigned int d = x1.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] pairs = np.zeros( (d,d), dtype=DTYPE )
    cdef unsigned int n, i, j

    # Compute one element of Pairs at a time
    for n in range( N ):
        for j in range( d ):
            for i in range( d ):
                pairs[i,j] += (x1[n,i] * x2[n,j] - pairs[i,j])/(n+1)
    return pairs

@cython.boundscheck(False) 
def Pairs2(np.ndarray[DTYPE_t, ndim=2] x1, np.ndarray[DTYPE_t, ndim=2] x2):
    """Compute E[x1 \ctimes x2]"""

    assert x1.dtype == DTYPE and x2.dtype == DTYPE

    cdef unsigned int N = x1.shape[0]
    cdef unsigned int d = x1.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] pairs = np.zeros( (d**2,d**2), dtype=DTYPE )
    cdef unsigned int n, i, j

    # Compute one element of Pairs at a time
    for n in range( N ):
        x1_ = np.outer( x1[n], x1[n] )
        x2_ = np.outer( x2[n], x2[n] )
        x12 = np.outer( x1_, x2_ )
        pairs += (x12 - pairs)/(n+1)
    return pairs

@cython.boundscheck(False) 
def PairsQ(np.ndarray[DTYPE_t, ndim=2] x, np.ndarray[DTYPE_t, ndim=1] q):
    """Compute E[x \ctimes x q(x)]"""

    cdef unsigned int N = x.shape[0]
    cdef unsigned int d = x.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] pairs = np.zeros( (d,d), dtype=DTYPE )
    cdef unsigned int n, i, j

    # Compute one element of Pairs at a time
    for n in range( N ):
        for j in range( d ):
            for i in range( d ):
                #pairs[i,j] += (q[n] * x[n,i] * x[n,j] - pairs[i,j])/(n+1)
                pairs[i,j] += q[n] * x[n,i] * x[n,j]
    return pairs

@cython.boundscheck(False) 
def Triples(np.ndarray[DTYPE_t, ndim=2] x1, np.ndarray[DTYPE_t, ndim=2]
        x2, np.ndarray[DTYPE_t, ndim=2] x3):
    """Compute E[x1 \ctimes x2 \ctimes x3 ]"""
    assert x1.dtype == DTYPE and x2.dtype == DTYPE and x3.dtype == DTYPE

    cdef unsigned int N = x1.shape[0]
    cdef unsigned int d = x1.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=3] triples = np.zeros( (d,d,d), dtype=DTYPE )
    cdef unsigned int n, i, j, k

    # Compute one element of Triples at a time
    for n in range( N ):
        for k in range( d ):
            for j in range( d ):
                for i in range( d ):
                    triples[i,j,k] += (x1[n,i] * x2[n,j] * x3[n,k] - triples[i,j,k])/(n+1)
    return triples

@cython.boundscheck(False) 
def TriplesQ(np.ndarray[DTYPE_t, ndim=2] x, np.ndarray[DTYPE_t, ndim=1] q):
    """Compute E[x1 \ctimes x2 \ctimes x3 ]"""

    cdef unsigned int N = x.shape[0]
    cdef unsigned int d = x.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=3] triples = np.zeros( (d,d,d), dtype=DTYPE )
    cdef unsigned int n, i, j, k

    # Compute one element of Triples at a time
    for n in range( N ):
        for k in range( d ):
            for j in range( d ):
                for i in range( d ):
                    triples[i,j,k] += q[n] * x[n,i] * x[n,j] * x[n,k] 
    return triples

@cython.boundscheck(False) 
def TriplesP(np.ndarray[DTYPE_t, ndim=2] x1, np.ndarray[DTYPE_t, ndim=2]
        x2, np.ndarray[DTYPE_t, ndim=2] x3, np.ndarray[DTYPE_t, ndim=1] theta):
    """Compute E[x1 \ctimes x2 \ctimes x3 ]"""
    assert x1.dtype == DTYPE and x2.dtype == DTYPE and x3.dtype == DTYPE

    cdef unsigned int N = x1.shape[0]
    cdef unsigned int d = x1.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] triples = np.zeros( (d,d), dtype=DTYPE )
    cdef DTYPE_t y
    cdef unsigned int n, i, j, k

    # Compute one element of Triples at a time
    for n in range( N ):
        y = 0
        for k in range(d):
            y += x3[n,k] * theta[k]
        for j in range( d ):
            for i in range( d ):
                triples[i,j] += (x1[n,i] * x2[n,j] * y - triples[i,j])/(n+1)
    return triples

@cython.boundscheck(False) 
def TriplesPQ(np.ndarray[DTYPE_t, ndim=2] x, np.ndarray[DTYPE_t, ndim=1] q, np.ndarray[DTYPE_t, ndim=1] theta):
    """Compute E[x \ctimes x \ctimes x q(x)]"""

    cdef unsigned int N = x.shape[0]
    cdef unsigned int d = x.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] triples = np.zeros( (d,d), dtype=DTYPE )
    cdef DTYPE_t y
    cdef unsigned int n, i, j, k

    # Compute one element of Triples at a time
    for n in range( N ):
        y = 0
        for k in range(d):
            y += x[n,k] * theta[k]

        for j in range( d ):
            for i in range( d ):
                triples[i,j] += (q[n] * x[n,i] * x[n,j] * y - triples[i,j])/(n+1)
    return triples

def apply_shuffle( np.ndarray[DTYPE_t, ndim=2] X, np.ndarray[LONG_t, ndim=1] perm ):
    assert X.dtype == DTYPE 

    cdef unsigned int N = X.shape[0]
    cdef unsigned int d = X.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=1] buf = np.zeros( (d,), dtype=DTYPE )
    for i in range( N-1 ):
        j = perm[i]
        if j != 0:
            buf = X[i]
            X[i] = X[i+j]
            X[i+j] = buf

    return X

@cython.boundscheck(False) 
def xMy(np.ndarray[DTYPE_t, ndim=2] M, np.ndarray[DTYPE_t, ndim=2] X, np.ndarray[DTYPE_t, ndim=2] Y):
    """Compute x^i M y^i"""

    assert X.shape[0] == Y.shape[0]
    assert X.shape[1] == Y.shape[1] and X.shape[1] == M.shape[0]

    cdef unsigned int N = X.shape[0]
    cdef unsigned int d = X.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=1] result = np.zeros( N, dtype=DTYPE )
    cdef unsigned int n, i, j

    # Compute one element of Pairs at a time
    for n in range( N ):
        for j in range( d ):
            for i in range( d ):
                result[n] += (M[i,j] * X[n,i] * Y[n,j] )
    return result

@cython.boundscheck(False) 
def Txyz(np.ndarray[DTYPE_t, ndim=3] T, np.ndarray[DTYPE_t, ndim=2] X, np.ndarray[DTYPE_t, ndim=2] Y, np.ndarray[DTYPE_t, ndim=2] Z):
    """Compute T(x^i, y^i, z^i)"""

    assert X.shape[0] == Y.shape[0] and X.shape[0] == Z.shape[0]
    assert X.shape[1] == Y.shape[1] and X.shape[1] == Z.shape[1] and X.shape[1] == T.shape[0]

    cdef unsigned int N = X.shape[0]
    cdef unsigned int d = X.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=1] result = np.zeros( N, dtype=DTYPE )
    cdef unsigned int n, i, j

    # Compute one element of Pairs at a time
    for n in range( N ):
        for k in range( d ):
            for j in range( d ):
                for i in range( d ):
                    result[n] += (T[i,j,k] * X[n,i] * Y[n,j] * Z[n,k] )
    return result

