#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import numpy as np
from numpy import zeros, diag, array, sqrt
from numpy.linalg import norm
import itertools as it
from munkres import Munkres 
from scipy.linalg import svd, svdvals, norm, eigvals, eig

from sktensor import ktensor 

from nose import with_setup
def set_random_seed():
    """
    Setup hook for test functions
    """
    np.random.seed(42)

def is_square(X):
    return all( dim == X.shape[0] for dim in X.shape )

def symmetric_skew(X):
    """
    A measure of how assymmetric the matrix/tensor is:
    |X_{ij} - X_{pi(ij)}|
    """
    assert is_square(X)

    D, R = X.shape[0], len(X.shape)
    err, cnt = 0, 0
    for idx in it.combinations_with_replacement( xrange(D), R ):
        err_, cnt_ = 0, 0
        for idx_ in set(it.permutations(idx)):
            err_ += abs(X[idx] - X[idx_])
            cnt_ += 1
        err += err_ / cnt_
        cnt += 1
    return err/cnt

def test_symmetric_skew():
    d = 3
    X = np.eye(3)
    assert np.allclose( symmetric_skew(X), 0. )
    X[0,1] = 0.5
    assert np.allclose( symmetric_skew(X), 0.5 )
    X[1,0] = 0.5
    assert np.allclose( symmetric_skew(X), 0. )

def column_norm(X):
    return np.sqrt(X**2).sum(0)
def row_norm(X):
    return np.sqrt(X**2).sum(1)

def normalize(X):
    return X / norm(X)

def normalize_columns(X):
    N, D = X.shape
    x = column_norm(X)
    return X / np.vstack([x for _ in xrange(N)])

def normalize_rows(X):
    N, D = X.shape
    x = row_norm(X)
    return X / np.vstack([x for _ in xrange(D)]).T

def match_rows(X, Y):
    """
    Permute the rows of _X_ to minimize error with Y
    @params X numpy.array - input matrix
    @params Y numpy.array - comparison matrix
    @return numpy.array - X with permuted rows
    """
    n, d = X.shape
    n_, d_ = Y.shape
    assert n == n_ and d == d_

    # Create a weight matrix to compare the two
    W = zeros((n, n))
    for i, j in it.product(xrange(n), xrange(n)):
        # Cost of 'assigning' j to i.
        W[i, j] = norm(X[j] - Y[i])

    matching = Munkres().compute( W )
    matching.sort()
    _, rowp = zip(*matching)
    rowp = array( rowp )
    # Permute the rows of B according to Bi
    X_ = X[ rowp ]

    return X_

@with_setup(set_random_seed)
def test_match_rows():
    k, d = 3, 5
    Y = np.random.rand(k, d)
    X = Y[[2,0,1]]
    X_ = match_rows(X, Y)
    assert np.allclose(X_, Y)

def match_columns(X, Y):
    """
    Permute the columns of _X_ to minimize error with Y
    @params X numpy.array - input matrix
    @params Y numpy.array - comparison matrix
    @return numpy.array - X with permuted columns
    """

    return match_rows(X.T, Y.T).T

@with_setup(set_random_seed)
def test_match_columns():
    k, d = 3, 5
    Y = np.random.rand(k, d)
    X = Y[:,[2,0,1,4,3]]
    X_ = match_columns(X, Y)
    assert np.allclose(X_, Y)

def match_rows_sign(X, Y):
    """
    Permute the rows of _X_ to minimize error with Y, ignoring signs of the columns
    @params X numpy.array - input matrix
    @params Y numpy.array - comparison matrix
    @return numpy.array - X with permuted rows
    """
    n, d = X.shape
    n_, d_ = Y.shape
    assert n == n_ and d == d_

    # Create a weight matrix to compare the two
    W = zeros((n, n))
    for i, j in it.product(xrange(n), xrange(n)):
        # Cost of 'assigning' j to i.
        W[i, j] = min(norm(X[j] - Y[i]), norm(X[j] + Y[i]) )

    matching = Munkres().compute( W )
    matching.sort()
    _, rowp = zip(*matching)
    rowp = array( rowp )
    # Permute the rows of B according to Bi
    X_ = X[ rowp ]
    # Change signs to minimize loss
    for row in xrange(n):
        if norm(X_[row] + Y[row]) < norm(X_[row] - Y[row]):
            X_[row] *= -1

    return X_

@with_setup(set_random_seed)
def test_match_rows_sign():
    k, d = 3, 5
    Y = np.random.rand(k, d)
    X = Y[[2,0,1]]
    X[1,:] *= -1
    X_ = match_rows_sign(X, Y)
    assert np.allclose(X_, Y)


def match_columns_sign(X, Y):
    """
    Permute the columns of _X_ to minimize error with Y
    @params X numpy.array - input matrix
    @params Y numpy.array - comparison matrix
    @return numpy.array - X with permuted columns
    """

    return match_rows_sign(X.T, Y.T).T

@with_setup(set_random_seed)
def test_match_columns_sign():
    k, d = 3, 5
    Y = np.random.rand(k, d)
    X = Y[:,[2,0,1,4,3]]
    X[:, 1] *= -1
    X[:, 3] *= -1
    X_ = match_columns_sign(X, Y)
    assert np.allclose(X_, Y)

def symmetrize(X):
    """
    Symmeterize X by taking the average over index swaps
    """

    def to_einstr(ls):
        return "".join([chr(ord('a') + l) for l in ls])

    D, R = X.shape[0], len(X.shape)
    X_ = np.zeros(X.shape)
    for new_order in it.permutations(xrange(R)):
        # Permute axes with einsum
        X_ += np.einsum('%s->%s'%(to_einstr(range(R)), to_einstr(new_order)), X) / np.math.factorial(R)

    return X_

def test_symmetrize():
    M = np.array([[1.,1.], [0.,0.]])
    M1 = symmetrize(M)
    assert np.allclose(M1, np.array([[1.,0.5], [0.5,0.]]))
    M2 = symmetrize(M1)
    assert np.allclose(M2, np.array([[1.,0.5], [0.5,0.]]))

def mrank( x, eps=1e-12 ):
    """Matrix rank"""
    d = svdvals( x )
    return len( [v for v in d if abs(v) > eps ] ) 

def tensorify(a, b, c):
    a = np.atleast_2d(a).T
    b = np.atleast_2d(b).T
    c = np.atleast_2d(c).T

    return ktensor([a,b,c]).totensor()

def svdk( X, k ):
    """Top-k SVD decomposition"""
    U, D, Vt = svd( X, full_matrices=False )
    return U[:, :k], D[:k], Vt[:k, :]

def approxk( X, k ):
    """Best k rank approximation of X"""
    U, D, Vt = svdk( X, k )
    return U.dot( diag( D ) ).dot( Vt )


def get_whitener( A, k ):
    """Return the matrix W that whitens A, i.e. W^T A W = I. Assumes A
    is k-rank"""

    U, D, _ = svdk(A, k)
    Ds = sqrt(D)
    Di = 1./Ds
    return U.dot(diag(Di)), U.dot(diag(Ds))


