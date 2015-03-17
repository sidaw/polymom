#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Representing lu factorization as a polynomial
"""

import scipy as sc
import scipy.linalg
import numpy as np
import sympy as sp
from numpy import zeros, array, triu_indices_from
from sympy import xring, RR, symbols
from util import row_normalize
import BorderBasis as BB

import ipdb

def lu_factorize(A):
    """
    Solves LU of A using polymom
    Assumes that diagonal elements of L are 1
    """
    d, d_ = A.shape
    assert d == d_ # Only triangular for now

    l_vars = [(i,j) for i in xrange(d) for j in xrange(i)]
    u_vars = [(i,j) for i in xrange(d) for j in xrange(i, d)]
    syms = ["l_%d%d"% (i+1,j+1) for i, j in l_vars] + ["u_%d%d"% (i+1,j+1) for i, j in u_vars]
    R, syms = xring(",".join(syms), RR)
    l_vars_ = dict(zip(l_vars, syms[:len(l_vars)]))
    u_vars_ = dict(zip(u_vars, syms[len(l_vars):]))

    # Construct equations
    def eqn(i,j):
        return sum(l_vars_.get((i,k), 1 if i == k else 0) * u_vars_.get((k,j), 0) for k in xrange(min(i,j)+1))
    I = [eqn(i,j) - A[i,j] for i in xrange(d) for j in xrange(d)]

    B = BB.BorderBasisFactory(1e-5).generate(R,I)
    print B
    V = B.zeros()
    assert len(V) == 1
    V = V[0]
    L = np.eye(d)
    U = np.zeros((d,d))
    for idx, (i, j) in enumerate(l_vars):
        L[i, j] = V[idx]

    for idx, (i, j) in enumerate(u_vars):
        U[i, j] = V[len(l_vars)+idx]

    return L, U

def test_lu_factorize(d=3):
    A = np.random.randn(d,d)
    L_, U_ = lu_factorize(A)
    assert np.allclose(L_.dot(U_), A)


def ll_factorize(A):
    """
    Solves LL of A using polymom
    Assumes that diagonal elements of L are 1
    M = L L^T; M_{ij} = \sum{k=1}^{min{i,j}} L_{ik} L_{jk}
    """
    d, d_ = A.shape
    assert d == d_ # Only triangular for now

    l_vars = [(i,j) for i in xrange(d) for j in xrange(i+1)]
    syms = ["l_%d%d"% (i+1,j+1) for i, j in l_vars]
    R, syms = xring(",".join(syms), RR)
    l_vars_ = dict(zip(l_vars, syms[:len(l_vars)]))

    # Construct equations
    def eqn(i,j):
        return sum(l_vars_.get((i,k), 1 if i == k else 0) * l_vars_.get((j,k), 0) for k in xrange(min(i,j)+1))
    I = [eqn(i,j) - A[i,j] for i in xrange(d) for j in xrange(i+1)]
    I += [l_vars_.get((0,0)) - np.sqrt(A[0,0])]

    return R, I

    B = BB.BorderBasisFactory(1e-5).generate(R,I)
    print B
    V = B.zeros()
    assert len(V) == 1
    V = V[0]
    L = zeros(d,d)
    for idx, (i, j) in enumerate(l_vars):
        L[i, j] = V[idx]

    return L


