#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
"""

import numpy as np

import ipdb

from tensor_power_method import candecomp
from sktensor import ktensor, dtensor, khatrirao
from util import Triples, mrank, approxk, svdk, symmetrize, symmetric_skew
from models.MixtureModel import MixtureModel
import scipy as sc
from scipy import diag, sqrt
from scipy.linalg import norm, svdvals, eig, eigvals, pinv, cholesky, inv
from util import closest_permuted_matrix, \
        closest_permuted_vector, column_aerr, column_rerr

from nose import with_setup

def set_random_seed():
    """
    Setup hook for test functions
    """
    sc.random.seed(42)

def get_moments(xs, k):
    n, d, v = xs.shape
    assert d >= k
    assert v >= 3

    xs1, xs2, xs3 = xs[:,:,0], xs[:,:,1], xs[:,:,2]

    m1 = (xs1.sum(0) / n).reshape(d,1)
    M2 = symmetrize(xs1.T.dot(xs2) / n)
    M3 = symmetrize(Triples(xs1, xs2, xs3))

    return m1, M2, M3

def get_whitener( A, k ):
    """Return the matrix W that whitens A, i.e. W^T A W = I. Assumes A
    is k-rank"""

    U, D, V = svdk(A, k)
    Ds = sqrt(D)
    Di = 1./Ds
    return U.dot(diag(Di)), U.dot(diag(Ds))

def find_means(X, K):
    """
    Use data X to solve for mean and std.
    """
    M1, M2, M3 = get_moments(X, K)
    return solve_with_moments(M1, M2, M3, K)

def solve_with_moments(m1, M2, M3, K):
    """
    Whiten and unwhiten appropriately
    """

    assert symmetric_skew(M2) < 1e-2
    assert symmetric_skew(M3) < 1e-2

    W, Wt = get_whitener( M2, K )
    M3_ = sc.einsum( 'ijk,ia,jb,kc->abc', M3, W, W, W )

    #print "M3", M3
    pi_, M_, _, _ = candecomp(M3_, K)
    #print "mu", M_
    mu = Wt.dot(M_.dot(diag(pi_)))
    return mu

@with_setup(set_random_seed)
def test_mom():
    """Test the whether spectral can recover from simple mixture of 3 identical"""
    model = MixtureModel.generate(2, 2, dirichlet_scale = 0.9)
    k, d, M, w = model["k"], model["d"], model["M"], model["w"]
    N = 1e5
    print "M", M

    xs = model.sample(N)
    print model.exact_moments(model.observed_monomials(3))
    print model.empirical_moments(xs, model.observed_monomials(3))

    M_ = find_means(xs, k)
    print "M_", M_

    M_ = closest_permuted_matrix( M.T, M_.T ).T
    print "M", M
    print "M_", M_

    print norm( M - M_ )/norm(M)

    assert( norm( M - M_ )/norm(M) < 1e-1 )
