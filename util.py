#!/usr/bin/env python2.7
"""
Various utility methods
"""
import numpy as np
import scipy as sc 
import operator
from itertools import chain
from sympy import grevlex
from numpy import array, zeros, diag, sqrt
from numpy.linalg import eig, inv, svd
import scipy.sparse
import ipdb
from munkres import Munkres
import sys

eps = 1e-15

def norm(arr):
    """
    Compute the sparse norm
    """
    if isinstance(arr, sc.sparse.base.spmatrix):
        return sqrt((arr.data**2).sum())
    else:
        return sqrt((arr**2).sum())


def tuple_add(t1, t2):
    """Elementwise addition of two tuples"""
    return tuple( t1[i] + t2[i] for i in xrange(len(t1)) )

def tuple_diff(t1, t2):
    """Elementwise addition of two tuples"""
    return tuple( t1[i] - t2[i] for i in xrange(len(t1)) )

def tuple_min(t1, t2):
    """Return the entry-wise minimum of the two tuples"""
    return tuple(min(a, b) for a, b in zip(t1, t2))

def tuple_max(t1, t2):
    """Return the entry-wise maximum of the two tuples"""
    return tuple(max(a, b) for a, b in zip(t1, t2))

def tuple_incr(t1, idx, val=1):
    """Return a tuple with the index idx incremented by val"""
    return t1[:idx] + (t1[idx]+val,) + t1[idx+1:]

def tuple_border(t):
    """Return the border of a tuple"""
    return [tuple_incr(t, i) for i in xrange(len(t))]

def tuple_subs(t1, t2):
    """
    Does t1_i > t2_i for all i? 
    """
    d = tuple_diff(t1, t2)
    if any(i < 0 for i in d):
        return False
    else:
        return True

def nonzeros(lst):
    """Return non-zero indices of a list"""
    return (i for i in xrange(len(lst)) if lst[i] > 0)

def first(iterable, default=None, key=None):
    """
    Return the first element in the iterable
    """
    if key is None:
        for el in iterable:
            return el
    else:
        for key_, el in iterable:
            if key == key_:
                return el
    return default

def prod(iterable):
    """Get the product of elements in the iterable"""
    return reduce(operator.mul, iterable, 1)

def to_syms(R, *monoms):
    """
    Get the symbols of an ideal I
    """
    return [prod(R(R.symbols[i])**j
                for (i, j) in enumerate(monom))
                    for monom in monoms]

def smaller_elements_(lst, idx = 0, order=grevlex):
    """
    Returns all elements smaller than the item in lst
    """
    assert order == grevlex
    if not isinstance(lst, list): lst = list(lst)
    if idx == 0: yield tuple(lst)
    if idx == len(lst)-1:
        yield tuple(lst)
        return

    tmp, tmp_ = lst[idx], lst[idx+1]
    while lst[idx] > 0:
        lst[idx] -= 1
        lst[idx+1] += 1
        for elem in smaller_elements_(lst, idx+1, order): yield elem
    lst[idx], lst[idx+1] = tmp, tmp_

def smaller_elements(lst, grade=None, idx = 0, order=grevlex):
    if not isinstance(lst, list): lst = list(lst)
    while True:
        for elem in smaller_elements_(lst, 0, order): yield elem
        # Remove one from the largest element
        for i in xrange(len(lst)-1, -1, -1):
            if lst[i] > 0: lst[i] -= 1; break
        else: break

def dominated_elements(lst, idx = 0):
    """
    Iterates over all elements that are dominated by the input list.
    For example, (2,1) returns [(2,1), (2,0), (1,1), (1,0), (0,0), (0,1)]
    """
    # Stupid check
    if not isinstance(lst, list): lst = list(lst)
    if idx == len(lst): yield tuple(lst)
    else:
        tmp = lst[idx]
        # For each value of this index, update other values
        while lst[idx] >= 0:
            for elem in dominated_elements(lst, idx+1): yield elem
            lst[idx] -= 1
        lst[idx] = tmp

def test_dominated_elements():
    """Simple test of generating dominated elements"""
    L = list(dominated_elements((2,1)))
    assert L == [(2,1), (2,0), (1,1), (1,0), (0,1), (0,0)]

def support(fs, order=grevlex):
    """
    Get the terms spanned by support of
    f_1, ... f_n
    """
    O = set(chain.from_iterable(f.monoms() for f in fs))
    return sorted(O, key=order, reverse=True)

def order_ideal(fs, order=grevlex):
    """
    Return the order ideal spanned by these monomials.
    """
    O = set([])
    for t in support(fs, order):
        if t not in O:
            O.update(dominated_elements(list(t)))
    return sorted(O, key=grevlex, reverse=True)

def lt(arr, tau=0):
    """
    Get the leading term of arr > tau
    """
    if arr.ndim == 1:
        idxs, = arr.nonzero()
    elif arr.ndim == 2:
        assert arr.shape[0] == 1
        idxs = zip(*arr.nonzero())
    else:
        raise Exception("Array of unexpected size: " + arr.ndim)
    for idx in idxs:
        elem = arr[idx]
        if abs(elem) > tau:
            if arr.ndim == 1:
                return idx, elem
            elif arr.ndim == 2:
                return idx[1], elem
    return 0, arr[0]

def lm(arr, tau=0):
    """Returns leading monomial"""
    return lt(arr, tau)[0]

def lc(arr, tau=0):
    """Returns leading coefficient"""
    return lt(arr, tau)[1]

def lt_normalize(R, tau=0):
    """
    Normalize to have the max term be 1
    """
    for i in xrange(R.shape[0]):
        R[i] /= lc(R[i], tau)
    return R

def row_normalize(R, tau = eps):
    """
    Normalize rows to have unit norm
    """
    for r in R:
        li = norm(r)
        if li < tau:
            r[:] = 0
        else:
            r /= li
    return R

def row_reduce(R, tau = eps):
    """
    Zero all rows with leading term from below
    """
    nrows, _ = R.shape
    for i in xrange(nrows-1, 0, -1):
        k, v = lt(R[i,:], tau)
        if v > tau:
            for j in xrange(i):
                R[j, :] -= R[i,:] * R[j,k] / R[i,k]
        else:
            R[i, :] = 0

    return R

def srref(A, tau=eps):
    """
    Compute the stabilized row reduced echelon form.
    """
    # TODO: Make sparse compatible
    if isinstance(A, sc.sparse.base.spmatrix):
        A = A.todense()
    A = array(A)
    m, n = A.shape

    Q = []
    R = zeros((min(m,n), n)) # Rows

    for i, ai in enumerate(A.T):
        # Remove any contribution from previous rows
        for j, qj in enumerate(Q):
            R[j, i] = ai.dot(qj)
            ai -= ai.dot(qj) * qj
        li = norm(ai)
        if li > tau:
            assert len(Q) < min(m,n)
            # Add a new column to Q
            Q.append(ai / li)
            # And write down the contribution
            R[len(Q)-1, i] = li

    # Convert to reduced row echelon form
    row_reduce(R, tau)

    # row_normalize
    row_normalize(R, tau)

    return array(Q).T, R

def test_srref():
    W = np.matrix([[  0.000e+00,   0.000e+00],
        [  0.000e+00,   0.000e+00],
        [  0.000e+00,   0.000e+00],
        [  5.020e-17,   1.180e-16],
        [ -4.908e-01,   6.525e-01],
        [ -8.105e-01,  -9.878e-02],
        [  0.000e+00,   0.000e+00],
        [  0.000e+00,   0.000e+00],
        [  3.197e-01,   7.513e-01]]).T
    return srref(W)

def simultaneously_diagonalize(Ms):
    """
    Simultaneously diagonalize a set of matrices.
    * Currently uses a crappy "diagonalize one and use for the rest"
      method.
    TODO: Use QR1JD.
    """
    it = iter(Ms)
    M = it.next()
    l, R = eig(M)
    Ri = inv(R)
    L = [l]
    for M in it:
        l = diag(Ri.dot(M).dot(R))
        L.append(l)
    return L, R

def truncated_svd(M, epsilon=eps):
    """
    Computed the truncated version of M from SVD
    """
    U, S, V = svd(M)
    S = S[abs(S) > epsilon]
    return U[:, :len(S)], S, V[:len(S),:]

def closest_permuted_vector( a, b ):
    """Find a permutation of b that matches a most closely (i.e. min |A
    - B|_2)"""

    # The elements of a and b form a weighted bipartite graph. We need
    # to find their minimal matching.
    assert( a.shape == b.shape )
    n, = a.shape

    W = sc.zeros( (n, n) )
    for i in xrange( n ):
        for j in xrange( n ):
            W[i, j] = (a[i] - b[j])**2

    m = Munkres()
    matching = m.compute( W )
    matching.sort()
    _, bi = zip(*matching)
    return b[array(bi)]

def closest_permuted_matrix( A, B ):
    """Find a _row_ permutation of B that matches A most closely (i.e. min |A
    - B|_F)"""

    # The rows of A and B form a weighted bipartite graph. The weights
    # are computed using the vector_matching algorithm.
    # We need to find their minimal matching.
    assert( A.shape == B.shape )

    n, _ = A.shape
    m = Munkres()

    # Create the weight matrix
    W = sc.zeros( (n, n) )
    for i in xrange( n ):
        for j in xrange( n ):
            # Best matching between A and B
            W[i, j] = norm(A[i] - B[j])
        
    matching = m.compute( W )
    matching.sort()
    _, rowp = zip(*matching)
    rowp = array( rowp )
    # Permute the rows of B according to Bi
    B_ = B[ rowp ]

    return B_

def save_example(R, I, out=sys.stdout):
    out.write(",".join(map(str, R.symbols)) +"\n")
    for i in I:
        out.write(str(i) + "\n")

