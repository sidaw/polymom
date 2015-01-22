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

    m, n = A.shape

    Q = zeros((m, 0))
    R = zeros((m, 0))

    for i in xrange(n):
        ai = A[:,i]
        qi = vert(ai) - vert(sum(ai.dot(qj) * qj for qj in Q.T))
        li = norm(qi)

        ri = zeros(m)
        print "li", i, li
        if li > tau:
            Q = hstack((Q, 1.0/li * qi))
            for j in xrange(Q.shape[1]-1):
                ri[j, 0] = li * ai.dot(col(Q,j)) 
            ri[Q.shape[1]-1, 0] = li
        R = hstack((R, vert(ri)))

    # Clear out columns
    for i in xrange(m-1,-1,-1):
        pivot = first(nonzero(R[i,:]))
        for j in xrange(i):
            # Get pivot
            R[j,:] = R[i,:] * R[j,pivot] / R[i,pivot]

    return Q, R    

