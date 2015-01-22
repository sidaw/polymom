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

    R = sp.matrix2numpy(sp.Matrix(A).rref(iszerofunc = lambda v : v < tau)[0], dtype=np.double)
    # Row normalize
    for i in xrange(R.shape[0]):
        R[i,:] /= norm(R[i,:])

    return R    

def test_reduced_row_echelon():
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
