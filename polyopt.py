# coding: utf-8
"""
Polynomial optimization routines
"""

import numpy as np
import itertools as it
from util import tuple_add

from cvxopt import matrix, solvers
import sympy as sp
import operator as op

from numpy.linalg import eig, inv
from numpy import diag

solvers.options['show_progress'] = True

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
    return sp.poly(ret)

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

order_monomial = sp.polys.orderings.monomial_key

def order_monomials(monoms, ordering='grlex'):
    return map(op.itemgetter(1), sorted(map(order_monomial(ordering), monoms)))

def monom_to_sym(monom, *syms):
    assert len(syms) == len(monom)
    term = 1.
    for sym, exp in zip(syms, monom):
        term *= sym ** exp
    return term

def get_leading_term(pol, *syms, **kwargs):
    """
    Get the leading term of the polynomial pol
    """
    ordering = kwargs.get('ordering', 'grlex')
    _, lt = max( map( order_monomial(ordering), pol.monoms() ) )
    if len(syms) == 0:
        return lt
    else:
        return monom_to_sym(lt, *syms)

def get_quotient_basis(g_basis, *syms):
    # - Get the leading terms
    lts = [ get_leading_term(b) for b in g_basis ]
    basis = set([])
    for lt in lts:
        terms = map(tuple,
                it.product(*[range(0,max(1,i)) for i in lt]))
        basis.update( terms )
    basis = order_monomials(basis)

    if len(syms) == 0:
        return lt
    else:
        return [monom_to_sym(b, *syms) for b in basis]

def linear_representation(p, basis):
    """
    Represent the polynomial p as a linear combination of the basis elements.
    """
    return np.array([p.coeff_monomial(b) for b in basis], dtype="float")

def construct_companion_matrix(g_basis, q_basis, sym):
    """
    Multiply q_basis by the sym and reduce using g_basis. The remainder is
    guaranteed to be in q_basis, and can be represented as such.
    """
    m = []
    for b in q_basis:
        _, r = g_basis.reduce(b * sym)
        m.append(linear_representation(r, q_basis))
    return np.array(m, dtype="float")

def test_simple_basis():
    """
    Test the creation of a quotient basis
    """
    x,y = sp.symbols('x,y')
    f1 = sp.poly( x**2 + y - 1, domain = 'C' )
    f2 = sp.poly( 2*x**2 + x*y + 2*y**2 + 1, domain = 'C' )
    I = [f1,f2]
    syms = [x,y]
    g_basis = sp.polys.groebner(I, *syms, order='grlex')
    q_basis = get_quotient_basis(I, *syms)
    assert len(q_basis) == 4

def solve_companion_matrix(I, *syms):
    """
    Solve the polynomial equations pol via the companion matrix method.
    """
    # 1) Find the grobner basis for I
    g_basis = sp.polys.groebner(I, *syms, order='grlex')
    print "Groebner basis:", g_basis
    # 2) Find the linear basis of C[X]/I
    q_basis = get_quotient_basis(g_basis, *syms)
    print "Quotient basis:", q_basis
    # 3) Construct companion matrices
    T = {}
    for sym in syms:
        T[sym] = construct_companion_matrix(g_basis, q_basis, sym)
    # 4) Solve
    l, R = eig(T[syms[0]])
    L = { key : diag(inv(R).dot(t).dot(R)) for key, t in T.items()}

    return L
