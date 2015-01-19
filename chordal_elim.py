#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Chordal elimination algorithm.
Cifuentes, D., & Parrilo, P. (2014, November 6). Exploiting chordal
structure in polynomial ideals: a Gröbner bases approach. arXiv [cs.SC].
Retrieved from http://arxiv.org/abs/1411.1745
"""

#from sympy import *
import sympy as sp
import numpy as np
from sympy import symbols, poly, ring, QQ, lex, Symbol, groebner
from util import nonzeros, first, prod

# Running examples
def example_simple():
    """
    Example 3.1 from paper. 4 variables, 4 equations, simple, method
    works
    """
    R, x0, x1, x2, x3 = ring('x0,x1,x2,x3', QQ, lex)
    I = [x0**4 - 1, x0**2 + x2, x1**2 + x2, x2**2 + x3]
    return R, I

def example_fail():
    """
    Example 3.2 from paper. 2 variables, 3 equations, method fails
    Failure mode: one of the ideal quotients is empty, which means that
    the variety could be the empty set.
    """
    R, x0, x1, x2 = ring('x0,x1,x2', QQ, lex)
    I = [x0 * x1 + 1, x1 + x2, x1*x2]
    return R, I

def example_order_preservation():
    """
    Example 3.4 from paper. 5 variables, 7 equations. Method fails
    unless the lex basis is re-added to the original basis
    """
    R, x0, x1, x2, x3, x4 = ring('x0,x1,x2,x3,x4', QQ, lex)
    I = [x0 - x2, x0 - x3, x1 - x3, x1 - x4, x2 - x3, x3 - x4, x2**2]
    return R, I

#####

def _syms(I):
    """
    Get the symbols of an ideal I
    """
    return I[0].ring.symbols

def to_syms(R, idxs):
    """
    Get the symbols of an ideal I
    """
    return [R.symbols[i] for i in idxs]

def get_clique(I, sym):
    """
    Find all symbols of I that appear with sym and are indexed
    greater than it.
    """

    # Find index.
    syms = _syms(I)
    idx = syms.index(sym) if type(sym) == Symbol else sym
    clique = set([idx])
    # Now, for every polynomial in I with this symbol, add all the other
    # variables.
    for f in I:
        degs = f.degrees()
        if degs[idx] > 0:
            clique.update(i for i in nonzeros(degs) if i > idx)

    return clique

def test_get_clique():
    """Test cliques for simple example I"""
    R, I = example_simple()
    x0, x1, x2, x3 = R.symbols
    assert list(get_clique(I, x0)) == [0, 2]
    assert list(get_clique(I, x1)) == [1, 2]
    assert list(get_clique(I, x2)) == [2, 3]
    assert list(get_clique(I, x3)) == [3]

def active_symbols(f):
    """Find the active symbols"""
    return set(nonzeros(f.degrees()))

def split_gens(I, X):
    """Split the generators of the ideal I into sets contained in K[X]
    and o.w."""
    J, K = [], []
    for f in I:
        if X.issuperset(active_symbols(f)):
            J.append(f)
        else:
            K.append(f)
    return J, K

def test_split_gens():
    """
    Test the split gens function
    """
    _, I = example_simple()
    X0 = get_clique(I, 0)
    J, K = split_gens(I, X0)
    assert I[0] in J
    assert I[1] in J
    assert I[2] not in J
    assert I[3] not in J

    assert I[0] not in K
    assert I[1] not in K
    assert I[2] in K
    assert I[3] in K

def as_poly(f, syms = None):
    if syms is None: syms = f.ring.symbols
    return poly(f.as_expr(), *syms, domain=f.ring.domain)

def as_polys(I, syms = None):
    return [as_poly(f, syms) for f in I]

def as_ring_expr(R, f):
    return R(f.as_expr())

def as_ring_exprs(R, I):
    return [as_ring_expr(R, f) for f in I]

def get_coefficient_term(R, f):
    """
    
    """
    terms = f.terms(R.order)
    syms = R.symbols
    LM, _ = terms[0]

    # get the first non-zero index - this is the "leading variable".
    leading_idx = first(nonzeros(LM))
    # Then, scan the rest of the terms for any thing other than this,
    # and add it to the polynomial.
    u = 0
    for monom, coeff in terms:
        if monom[leading_idx] != LM[leading_idx]: continue
        monom = monom[:leading_idx] + (0,) + monom[leading_idx+1:]
        u += coeff * prod(syms[var]**exp for (var, exp) in zip(xrange(len(monom)), monom))
    return R(u)

def test_get_coefficient_term():
    R, x, y, z = ring('x,y,z', QQ, lex)
    f = R(3 * x**2 * y * z + 2 * x**2 * y**2 + z + x * y * z)
    coeff = get_coefficient_term(R, f)
    assert coeff == R(2 * y**2 + 3 * y * z)

def eliminate(R, I, L=-1):
    """
    Inputs:
    $I$ - list[polynomial]: generators of ideal
    $L$ - integer: number of elimination ideals to obtain
    Output:
    $(I_L, [W_1, ..., W_L])$ - tuple[polynomial, list[polynomial]]: approximation of $elim_L(I)$
    * Implicitly construct "chordal graph" G
    """
    syms = R.symbols

    if L < 0:
        L = len(syms) - 1

    I = list(I) # copy
    W = []
    for l in xrange(L):
        # get clique X_l of "G"
        X_l = get_clique(I, l)
        print "X_l", X_l
        # split the I_l in to two sets of generators J_l, K_l
        J, K = split_gens(I, X_l)
        # Get lex grobner basis for J_l and append to J_l
        J = as_ring_exprs(R, groebner(as_polys(J), to_syms(R, X_l), domain=R.domain, order=R.order)) + J
        # Eliminate x_l from J_l
        print "J", J
        J, _ = split_gens(set(J), X_l.difference([l]))
        # TODO: eliminate x_l
        #J = as_ring_exprs(R, groebner(as_polys(J), to_syms(R, X_l), domain=R.domain, order=R.order, wrt='x%d'%l))[1:]
        print "J", J
        # Construct W_l from the coeff ring.
        W_ = [get_coefficient_term(R, f) for f in J] + K
        if 1 in W_: W_ = 1
        # Re construct the next I
        I = J + K
        W.append(W_)
        print "I_l, W_l", I, W_
    return I, W

def do_test(args):
    """
    Interactive test harness
    """
    pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser( description='' )
    parser.add_argument( '', type=str, help="" )
    parser.set_defaults(func=do_test)

    #subparsers = parser.add_subparsers()
    #command_parser = subparsers.add_parser('command', help='' )
    #command_parser.set_defaults(func=do_command)

    ARGS = parser.parse_args()
    ARGS.func(ARGS)
