#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Robust tensor power method as described in:
Anandkumar/Ge/Hsu/Kakade/Telgarsky, 2012.
"""

import numpy as np
from numpy import zeros, ones, eye, diag
from numpy.linalg import inv, norm, svd, eig
from numpy.random import rand, randn
import sktensor as ten
from sktensor import dtensor, khatrirao
import itertools as it

#from linalg import diagonalize
from util import orthogonal
from util import match_columns_sign, symmetric_skew, is_square, normalize_columns, normalize


from nose import with_setup
import ipdb

def set_random_seed():
    """
    Setup hook for test functions
    """
    np.random.seed(42)

def prerequisites( T, K ):
    """
    Does the tensor satisfy the prerequisites?
    (a) tensor should be symmetric (orthogonal)
    """
    if len(T.shape) != 3:
        return False
    elif not is_square(T):
        return False
    elif T.shape[0] < K:
        return False
    elif symmetric_skew(T) > 1e-1:
        return False
    else:
        return True

def deflate(T, lbda, v):
    v = np.atleast_2d(v).T
    return T - lbda * ten.ktensor([v,v,v]).totensor()

def find_max_eigenpair(T, outer_iterations = 10, inner_iterations = 100):
    """
    Run tensor power method (Algorithm 1 of Anandkumar/Ge/Hsu/Kakade/Telgarsky, 2012).
    """
    D = T.shape[0]
    eps = 1e-10

    best = (-np.inf, np.zeros(D))
    # Outer iterations
    for tau in xrange(outer_iterations):
        # (1) Draw a random initialization θ_t
        theta = normalize( randn( D ) )
        # Inner iterations
        for t in xrange(inner_iterations):
            # 2) Update θ ← T(I, θ, θ)/||T(I, θ, θ)||
            theta_ = normalize( T.ttv( (theta, theta), modes = (1,2) ) )
            if norm(theta - theta_) < eps:
                break
        # (3) Choose θ_t with max eigenvalue λ = T(θ, θ, θ)
        lbda = float( T.ttv( (theta, theta, theta), modes = (0,1,2) ) )
        epair = lbda, theta
        if epair[0] > best[0]:
            best = epair

    _, theta = best
    for t in xrange(inner_iterations):
        # 2) Update θ ← T(I, θ, θ)/||T(I, θ, θ)||
        theta = normalize( T.ttv( (theta, theta), modes = (1,2) ) )
    # (4) Update θ
    lbda = float(T.ttv( (theta, theta, theta), modes = (0,1,2) ))
    # (5) Return λ, θ
    return lbda, theta

def candecomp( T, K, outer_iterations = "k", inner_iterations = "k", **kwargs ):
    """
    Return the canonical rank K decomposition of a tensor T.
    @assumptions - T is symmetric orthogonal
    @params L - number of projections
    """
    assert prerequisites(T, K)
    if outer_iterations == "k":
        outer_iterations = 2*K
    else:
        assert int(outer_iterations)
    if inner_iterations == "k":
        inner_iterations = 10*int(np.ceil(np.log(K)))
    else:
        assert int(inner_iterations)

    T = dtensor(T)

    # (1) Outer loop - find distinct eigen vectors
    pi, W = [], []
    for _ in xrange(K):
        l, v = find_max_eigenpair(T, outer_iterations, inner_iterations)
        pi.append(l)
        W.append(v)
        T = deflate(T, l, v)
    pi, W = np.array(pi), np.array(W).T
    
    return pi, W, W, W

@with_setup(set_random_seed)
def test_candecomp():
    """
    Test if it works
    """
    d = 3
    pi = rand(d)
    A = orthogonal(3)
    T = ten.ktensor( [A, A, A], pi ).totensor()
    pi_, A_, B_, C_ = candecomp( T, d )
    T_ = ten.ktensor( [A_, B_, C_], pi_ ).totensor()

    pi_ = match_columns_sign(np.atleast_2d(pi_), np.atleast_2d(pi)).flatten()
    A_ = match_columns_sign(A_, A)

    assert np.allclose(pi, pi_)
    assert np.allclose(A, A_)

    assert np.allclose( T, T_ )

