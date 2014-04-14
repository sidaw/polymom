#!/usr/bin/env python2.7
# coding: utf-8
"""
Polynomial optimization routines
"""

import numpy as np
import itertools as it
from util import tuple_add

from cvxopt import matrix, solvers

def basis_transformation(max_degree, basis = "power"):
    """
    Returns a linear transformation (i.e. a matrix) from the power
    series basis (1, x, x^2, ...) to the specified basis.

    @param max_degree double - maximum degree of the power basis to be considered
    @param basis ('symmetric' | 'symmetric_irreducible') - basis to transform to 
    @returns matrix - the linear transform
    """

    if basis == "power":
        return np.eye(max_degree)
    else:
        raise NotImplementedError("no support for basis " + basis)

def construct_basis(pol):
    """
    Construct a basis for the polynomial
    """
    monoms = pol.monoms()
    max_degree = np.array(monoms).max(0)
    max_degree = [ int(np.ceil(deg/2.0)) for deg in max_degree ]

    basis = it.product(*(xrange(deg+1) for deg in max_degree))

    return list(basis)

def find_coeffs(pol):
    """Construct matrix of coefficients"""
    coeffs = pol.coeffs()
    monoms = pol.monoms()

    basis = construct_basis(pol)

    # Fill in the coefficients
    p = len(basis)
    A = np.zeros((p,p))
    for i in xrange(p):
        for j in xrange(p):
            basis_ = tuple_add(basis[i], basis[j])
            A[i,j] = coeffs[monoms.index(basis_)]
    return basis, A

def optimize_polynomial(pol):
    r"""
    Optimization polynomial by representing it as an SDP in the
    appropriate basis
    TODO: Support constraints

    \min_y c^\top y
           subject to M(y) \succeq 0
           y_1 = 1

    @param poly       - polynomial to optimize
    @param basis      - basis for the polynomial (can be 'power',
                        'symmetric', 'symmetric_irreducible')
    @returns (p*, y*) - the optimal value of the polynomial and
                        y* = E_\mu[b(x)].
    """

    coeffs = map(float, pol.coeffs())
    monoms = pol.monoms()
    # basis in sorted order from 1 to largest
    coeffs.reverse()

    max_degree = np.array(monoms).max(0)
    basis = construct_basis(pol)

    def idx(b):
        ret, multiplier = 0, 1
        for sym in xrange(len(max_degree)):
            ret += multiplier * b[sym]
            multiplier *= max_degree[sym]
        return ret


    # x^2 => x^1 x^1
    # x^4 => x^2 * x^2
    D = len(basis)
    B = 2 * D - 1 # y_0, y_1, y_2

    # Construct the matrix M_{ij} = y_{i+j} ≥ 0
    # Decomposes as \sum_{ij} \delta_{ij} y_{i+j}
    G = np.zeros( (B, D**2) ) # A p**2 matrix with p values
    for b1 in basis:
        for b2 in basis:
            G[idx(tuple_add(b1, b2)) , D * idx(b1)  + idx(b2)] += 1
    h = np.zeros((D, D))

    # Finally, we need to set y_1 = 1
    A = np.zeros((B, B))
    A[0,0] = 1
    b = np.zeros(B)
    b[0] = 1

    # Pad zeros
    while len(coeffs) < B:
        coeffs.append(0.)

    print coeffs
    print G
    print h
    print A
    print b

    return solvers.sdp(
            matrix(coeffs), 
            Gs = [matrix(G.T)],
            hs = [matrix(h)],
#            A  = matrix(A),
#            b  = matrix(b),
            )

