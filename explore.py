#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""

"""

from sympy import *
from polyopt import *
import ipdb

def spol(f,g):
    F, G = f.unify(g)
    domain, gens = F.domain, F.gens

    s = Poly(lcm(LM(f), LM(g)), *gens, domain=domain)
    tf = Poly(s/LT(f), *gens, domain=domain)
    tg = Poly(s/LT(g), *gens, domain=domain)

    return tf * f - tg * g

def rem(f, I):
    gens, domain = f.gens, f.domain
    ordering = polys.orderings.monomial_key(order='grevlex', gens=f.gens)
    # Sorted I 
    I = sorted(I, key = lambda i: ordering(LT(i)))
    m = len(I)
    coeffs = [0] * m

    while f != 0:
        for i_ in xrange(m):
            i = I[i_]
            q, r = div(LT(f), LT(i))
            if r == 0:
                f = f - Poly(q, *gens, domain=domain) * i
                coeffs[i_] += q
                break
        else:
            break
    
    return coeffs, f

def reduce_basis(I):
    gens, domain = I[0].gens, I[0].domain
    ordering = polys.orderings.monomial_key(order='grevlex', gens=gens)
    I = sorted(I, key = lambda i: ordering(LT(i)))
    i_ = 1
    while i_ < len(I):
        _, r = rem(I[i_], I[:i_])
        if r == 0:
            del I[i_]
        else:
            i_ += 1
    return I

def buchbergers(I):
    gens, domain = I[0].gens, I[0].domain
    ordering = polys.orderings.monomial_key(order='grevlex', gens=gens)
    J = I

    # Sorted I 
    while True:
        J = sorted(J, key = lambda i: ordering(LT(i)))
        m = len(J)
        lms = [LM(j) for j in J]
        print "J: ", m, lms

        for i, j in ((i,j) for i in xrange(m) for j in xrange(i, m)):
            s = spol(J[i], J[j])
            _, r = rem(s, J)
            if s != 0 and r != 0 and LM(r) not in lms:
                J.append(r)
                J = reduce_basis(J)
                break
        else:
            break
    # Reduce the basis

    return J

def linear_representation(p, basis):
    """
    Represent the polynomial p as a linear combination of the basis elements.
    """
    return np.array([p.coeff_monomial(b) for b in basis])

def construct_companion_matrix(g_basis, q_basis, sym):
    """
    Multiply q_basis by the sym and reduce using g_basis. The remainder is
    guaranteed to be in q_basis, and can be represented as such.
    """

    domain, gens = g_basis[0].domain, g_basis[0].gens

    m = []
    for b in q_basis:
        f = poly(b * sym, *gens, domain=domain)
        _, r = rem(f, g_basis)
        m.append(linear_representation(r, q_basis))
    return np.array(m)

def test_small():
    x, y = symbols('x,y')
    a, b = symbols('a,b')

    f = poly(x + y - a, x, y, domain=RR[a,b])
    g = poly(x * y - b, x, y, domain=RR[a,b])

    syms = [x, y]
    I = [f, g]

    gb = buchbergers(I)
    qb = get_quotient_basis(gb, *syms)
    print "qb", qb
    for sym in syms:
        M = construct_companion_matrix(gb, qb, sym) 
        print sym
        print M

def test_mom():
    x1, x2, y1, y2 = symbols('x1,x2,y1,y2')
    m, n = symbols('m,n')
    syms = [x1, x2, y1, y2]
    syms_ = [m, n]

    mx = poly(x1 + x2 - m, *syms, domain=QQ.frac_field(m,n))
    my = poly(y1 + y2 - n, *syms, domain=QQ.frac_field(m,n))
    mxx = poly(x1**2 + x2**2 - 1., *syms, domain=QQ.frac_field(m,n))
    mxy = poly(x1*y1 + x2*y2 - 0., *syms, domain=QQ.frac_field(m,n))

    I = [mx, my, mxx, mxy]
    print "Input ideal"
    for i, g in enumerate(I): pprint((i, g.as_expr()))

    print "Building Gröbner basis"
    gb = buchbergers(I)
    print "Gröbner Basis:"
    for i, g in enumerate(gb): pprint((i, g.as_expr()))
    qb = get_quotient_basis(gb, *syms)
    print "Quotient basis: "
    pprint(qb)

    print "Companion matrix:"
    for sym in syms:
        M = construct_companion_matrix(gb, qb, sym) 
        pprint(sym)
        print M
# 
# x, y = symbols('x,y')
# a, b = symbols('a,b')
# 
# f = poly(x + y - a, x, y, domain=RR[a,b])
# g = poly(x * y - b, x, y, domain=RR[a,b])
# 

test_mom()

