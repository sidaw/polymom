# coding: utf-8
"""
Polynomial optimization routines
"""

import numpy as np
import itertools as it
from util import tuple_add

from cvxopt import matrix, solvers
from sympy import symbols, poly

def by_evaluation(*pols, **args):
    r"""
    Creates a polynomial by evaluating each of pols with the arguments in **kwargs

    @param *pols polynomial list - list of polynomials to evaluate
    @param **kwargs symbol map - map of variable to value
    @return polynomial - is \sum (pol_i - pol_i(x))^2
    """
    ret = 0.
    for pol in pols:
        ret += (pol - pol.evalf(subs=args))**2
    return poly(ret)

def get_max_degree(pol):
    """
    Returns the largest degree of all monomials in the polynomial
    """
    return reduce( lambda deg, term: map(max, zip(deg, term)), pol.monoms())

def get_monom(pol, degrees):
    """
    Get a monomial term from the polynomial with the specified degree sequence.
    """
    return reduce(lambda acc, (sym, deg) : acc * sym**deg, zip(list(pol.free_symbols), degrees), 1)

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

    # basis in sorted order from 1 to largest
    # Get the largest monomial term
    max_degree = get_max_degree(pol)
    half_degree = map( lambda deg: int(np.ceil(deg/2.)), max_degree)

    # Generate basis
    basis = list(it.product( *[range(deg + 1) for deg in max_degree] ) )
    half_basis = list(it.product( *[range(deg + 1) for deg in half_degree] ) )
    N, M = len(basis), len(half_basis)

    # The coefficient term
    c = np.matrix([float(pol.nth(*elem)) for elem in basis]).T
    # The inequality term, enforcing sos-ness
    G = np.zeros( ( M**2, N ) )
    for i, j in it.product(xrange(M), xrange(M)):
        e1, e2 = half_basis[i], half_basis[j]
        monom = tuple_add(e1,e2)
        k = basis.index(monom)
        if k != -1:
            G[ M * i + j, k ] = 1
    h = np.zeros((M, M))

    # Finally, y_1 = 1
    A = np.zeros((1, N))
    A[0,0] = 1
    b = np.zeros(1)
    b[0] = 1

    sol = solvers.sdp(
            c = matrix(c),
            Gs = [matrix(-G)],
            hs = [matrix(h)],
            A  = matrix(A),
            b  = matrix(b),
            )
    assert sol['status'] == 'optimal'

    return dict([ (get_monom(pol,coeff), val) for coeff, val in zip(basis, list(sol['x'])) ] )

