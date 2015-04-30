#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
"""

import numpy as np

from tensor_power_method import candecomp
from sktensor import ktensor, dtensor, khatrirao
from util import Triples, mrank, approxk, svdk
from models.GaussianMixtures import GaussianMixtureModel
import scipy as sc
from scipy.linalg import norm, svdvals, eig, eigvals, pinv, cholesky
from util import closest_permuted_matrix, \
        closest_permuted_vector, column_aerr, column_rerr

from nose import with_setup
def set_random_seed():
    """
    Setup hook for test functions
    """
    np.random.seed(42)

def get_moments(xs, k):
    n, d = xs.shape
    assert d >= k

    m1 = (xs.sum(0) / n).reshape(d,1)
    m2 = xs.T.dot(xs) / n 
    U, S, _ = np.linalg.svd(m2 - m1.dot(m1.T))

    sigma2, v = S[-1], U[:,-1]

    m1 = (np.atleast_2d((xs - m1.T).dot(v)**2).T * xs ).sum(0) / n
    M1 = np.hstack(np.atleast_2d(m1).T for _ in range(d))
    #M1 = sigma2 * np.hstack( (m1 for _ in range(d)) )

    M2 = m2 - sigma2 * np.eye(d)
    M3 = Triples(xs, xs, xs)
    M3 -= ktensor([M1, np.eye(d), np.eye(d)]).totensor()
    M3 -= ktensor([np.eye(d), M1, np.eye(d)]).totensor()
    M3 -= ktensor([np.eye(d), np.eye(d), M1]).totensor()
    return m1, M2, M3

def get_whitener( A, k ):
    """Return the matrix W that whitens A, i.e. W^T A W = I. Assumes A
    is k-rank"""

    #assert( mrank( A ) == k )
    # Verify PSD
    e = eigvals( A )[:k].real
    if not (e >= 0).all():
      print "Warning: Not PSD"
      print e

    # If A is PSD
    U, _, _ = svdk( A, k )
    A2 = cholesky( U.T.dot( A ).dot( U ) )
    W, Wt = U.dot( pinv( A2 ) ), U.dot( A2 )
    
    return W, Wt

def find_means(X, K):
    """
    Use data X to solve for mean and std.
    """

    M1, M2, M3 = get_moments(X, K)
    return solve_with_moments(M1, M2, M3, K)

def solve_with_moments(M1, M2, M3, K):
    W, Wt = get_whitener( M2, K )
    M3_ = sc.einsum( 'ijk,ia,jb,kc->abc', M3, W, W, W )

    print "M3", M3
    pi_, M_, _, _ = candecomp(M3_, K)
    print "mu", M_
    mu = Wt.dot(M_.dot(np.diag(pi_)))
    return mu

@with_setup(set_random_seed)
def test_gaussian_mom():
    """Test the Gaussian EM on a small generated dataset"""
    fname = "gmm-3-10-0.7.npz"
    gmm = GaussianMixtureModel.generate( fname, 2, 3 )
    k, d, M, S, w = gmm.k, gmm.d, gmm.means, gmm.sigmas, gmm.weights
    N, n = 1e6, 1e6

    X = gmm.sample( N, n )

    M_ = find_means(X, k)
    print "M_", M_

    M_ = closest_permuted_matrix( M, M_ )
    print M, M_

    print norm( M - M_ )/norm(M)

    assert( norm( M - M_ )/norm(M) < 1e-1 )
