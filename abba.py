#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
The approximate border basis algorithm from:
Heldt, D., Kreuzer, M., Pokutta, S., & Poulisse, H. (2009). Approximate
computation of zero-dimensional polynomial ideals. Journal of Symbolic
Computation, 44(11), 1566–1591. doi:10.1016/j.jsc.2008.11.010
"""

import numpy as np
import sympy as sp
from numpy import array, zeros, atleast_2d, hstack
from numpy.linalg import norm

def vert(v):
    return atleast_2d(v).T

def col(M, n):
    return atleast_2d(M[:,n]).T

def row(M, n):
    return atleast_2d(M[n,:])

def reduced_row_echelon(A, tau):
    """
    a1, ..., an are columns of $A$. 
    This routine computes a reduced row echelon form with tolerance tau.
    """

    R = sp.matrix2numpy(sp.Matrix(A).rref(iszerofunc = lambda v : v < tau)[0],
            dtype=np.double)
    # Row normalize
    for i in xrange(R.shape[0]):
        R[i,:] /= norm(R[i,:])

    return R    

def test_reduced_row_echelon():
    """
    This example was taken from the paper:
    Subideal Border Bases: Martin Kreuzer, Henk Poulisse

    It seems to be incorrect because the right answer (C) is not in
    reduced row echelon form.
    """

    B = array([[0.0004, 0.6755, -0.5089, -0.5068, -0.1667],
               [0, -0.3812, -0.3735, -0.3812, 0.7548]])
    C = array([[0, 0.3812, 0.3735, 0.3812, -0.7548],
               [0, 0, 0.5754, 0.5811, -0.5754]])
    tau = 0.0001
    C_ = reduced_row_echelon(B, tau)
    print C
    print C_
    assert max(abs(C - C_).flatten()) < tau


    B = [[0.0004, 0.6755, -0.5089, -0.5068, -0.1667], [0, -0.3812, -0.3735, -0.3812, 0.7548]]

def example():
    """
    Example 4.12
    """
    R, x, y, z = ring('x,y,z', RR, lex)
    I = [0.130 * z**2  + 0.39 * y - 0.911 * z,
         0.242 * y * z - 0.97 * y,
         0.243 * x * z - 0.97 * y,
         0.242 * y**2  - 0.97 * y,
         0.243 * x * y - 0.97 * y,
         0.035 * x**5 - 0.284 * x**4 + 0.497 * x**3 + 0.284 * x**2 - 0.532 * x + 0.533 * y
         ]
    return R, I

def dominated_elements(lst, idx = 0):
    """
    Iterates over all elements that are dominated by the input list.
    For example, (2,1) returns [(2,1), (2,0), (1,1), (1,0), (0,0), (0,1)]
    """

    # Yield (a copy of) this element
    yield tuple(lst)

    # Update all subsequent indices
    for idx_ in xrange(idx, len(lst)):
        tmp = lst[idx]

        # Ticker down this index
        while lst[idx] > 0:
            lst[idx] -= 1
            for elem in count_down(lst, idx+1): yield elem
        lst[idx] = tmp

def get_linear_basis(*fs):
    """
    Get the order ideal corresponding to the terms spanned by support of
    f_1, ... f_n
    """
    O = set([])
    for f in fs:
        for m in f.monoms():
            O.update((dominated_elements(list(m))))
    return sorted(O)

def approx_unitary_basis(L, *I):
    """
    Construct a matrix with terms in L
    """
    M = zeros((len(L), len(I)))
    for i, f in enumerate(I):
        for monom, coeff in f.terms():
            M[L.index(monom), i] = coeff
    return M


def compute(R, I, delta):
    """
    Compute border basis for I = (f1, ..., fr) that is delta close.
    Uses IABBA algorithm (Corr 4.11)
    """

    # L is order ideal spanned by supp(f_i).
    L = get_linear_basis(*I)
    # Compute unitary basis V = {f1', ... , fr'}
    V = approx_unitary_basis(L, *I)

    # Compute approximate basis extension.
    # Expand order ideal $L$ with $W$
    # Update order ideal $O$
    # Expand order ideal $L$ with extension until border of $O$ matches.
    # Apply final reduction algorithm.




