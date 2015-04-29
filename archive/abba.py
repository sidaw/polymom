#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
The approximate border basis algorithm from:
Heldt, D., Kreuzer, M., Pokutta, S., & Poulisse, H. (2009). Approximate
computation of zero-dimensional polynomial ideals. Journal of Symbolic
Computation, 44(11), 1566–1591. doi:10.1016/j.jsc.2008.11.010
"""

import numpy as np
import sympy as sp
from numpy import array, zeros, atleast_2d, hstack, diag, sign
from numpy.linalg import norm, svd, qr

from sympy import ring, RR, lex, grevlex, pprint
from util import *
from itertools import chain

import ipdb

do_debug = False
eps = 1e-10

def problem_kids(M):
    return (abs(M[M.nonzero()]) < eps).any()

def vert(v):
    return atleast_2d(v).T

def col(M, n):
    return atleast_2d(M[:,n]).T

def row(M, n):
    return atleast_2d(M[n,:])

def sel(lst, idxs):
    return [lst[i] for i in idxs]

def difference(l1, l2):
    return set(l1).difference(l2)

def print_space(R, B, V):
    for v in V.dot(to_syms(R,*B)):
        pprint(v)

def coeff(B, v, term):
    return v[B.index(term)]

def lti(B, V):
    """
    Leading term ideal
    """
    return [B[lt(f)[0]] for f in V]

def srref_(A, tau = 1e-4):
    Q, R = np.linalg.qr(A)
    R = row_reduce(R, tau)
    R = row_normalize(R, tau)
    return Q, R

def rref_(A, tau = 1e-4):
    R = sp.matrix2numpy(sp.Matrix(A).rref(iszerofunc = lambda v : abs(v) < tau)[0],
            dtype=np.double)
    return row_normalize(R)

def rref(A, tau = 1e-4):
    """
    a1, ..., an are columns of $A$. 
    This routine computes a reduced row echelon form with tolerance tau.
    """

    #R = sp.matrix2numpy(sp.Matrix(A).rref(iszerofunc = lambda v : abs(v) < tau)[0],
    #        dtype=np.double)
    #return row_normalize(R)
    #R_ = rref_(A, tau) 
    #R = R_
    #_, R = srref_(A, tau)
    _, R = srref(A, tau)

    #assert np.allclose(R, R_)

    R[abs(R) < eps] = 0

    return R

def test_rref():
    """
    This example was taken from the paper:
    Subideal Border Bases: Martin Kreuzer, Henk Poulisse

    It seems to be incorrect because the right answer (C) is not in
    reduced row echelon form.
    """

    B = array([[0.0004, 0.6755, -0.5089, -0.5068, -0.1667],
               [0, -0.3812, -0.3735, -0.3812, 0.7548]])
    C = array([[0, 0.3812, 0.3735, 0.3812, -0.7548],
               [0, 0, 0.5754, 0.5811, -0.5754]])
    tau = 0.0001
    C_ = srref(B, tau)
    print C
    print C_
    assert max(abs(C - C_).flatten()) < tau


    B = [[0.0004, 0.6755, -0.5089, -0.5068, -0.1667], [0, -0.3812, -0.3735, -0.3812, 0.7548]]

def example_trivial():
    """
    My own example.
    """

    R, x, y = ring('x,y', RR, grevlex)

    I = [x**2 + y**2 - 2,
            x + y 
            ]

    return R, I

def example_simple():
    """
    Example 19 from KK
    """
    R, x, y = ring('x,y', RR, grevlex)
    I = [x**3 - x,
         y**3 - y,
         x**2*y - 0.5 * y - 0.5 * y**2,
         x*y - x - 0.5 * y + x**2 - 0.5 * y**2,
         x * y**2 - x - 0.5 * y + x**2 - 0.5 * y**2
         ]
    G = [x**2 + x * y - 0.5 * y**2 - x - 0.5 * y,
         y**3 - y,
         x*y**2 - x*y,
         x**2 * y - 0.5 * y**2 - 0.5 * y
         ]

    return R, I, G

def example_20():
    """
    Example 20 from KK
    """
    R, x, y, z = ring('x,y,z', RR, grevlex)
    I = [z**2 + 3 * y - 7 * z,
         y * z - 4 * y,
         x * z - 4 * y,
         y**2 - 4*y,
         x*y - 4 * y,
         x**5 - 8 * x**4 + 14 * x**4 + 8 * x**2 - 15 * x + 15 * y,
        ]
    G = [z**2 + 3 * y - 7 *z,
            y * z - 4 * y,
            x * z - 4 * y,
            y**2 - 4*y,
            x *y - 4 * y,
            x**2 * z - 16 * y,
            x **2 * y - 16 * y,
            x**3 * z - 64 * y,
            x**3 * y - 64 * y,
            x**4 * z - 256 * y,
            x**4 * y - 256 * y,
            x **5 - 8 * x **4 + 14 * x **4 + 8 * x **2 - 15 * x + 15 * y,
            ]

    return R, I, G

def example():
    """
    Example 4.12
    """
    R, x, y, z = ring('x,y,z', RR, grevlex)
    I = [0.130 * z**2  + 0.39 * y - 0.911 * z,
         0.242 * y * z - 0.97 * y,
         0.243 * x * z - 0.97 * y,
         0.242 * y**2  - 0.97 * y,
         0.243 * x * y - 0.97 * y,
         0.035 * x**5 - 0.284 * x**4 + 0.497 * x**3 + 0.284 * x**2 - 0.532 * x + 0.533 * y
         ]
    return R, I

def get_support_basis(fs, order=grevlex):
    """
    Get the terms spanned by support of
    f_1, ... f_n
    """
    O = set(chain.from_iterable(f.monoms() for f in fs))
    return sorted(O, key=grevlex, reverse=True)

def test_get_support_basis():
    _, x, y = ring('x,y', RR, grevlex)
    f = x**2 + x*y + y
    g = x**2 + x*y + x
    b = get_support_basis([f,g])
    assert b == [(2,0), (1,1), (1,0), (0,1)]

def get_order_basis(fs, order=grevlex):
    """
    Get the order ideal corresponding to the terms spanned by support of
    f_1, ... f_n
    """
    O = set(chain.from_iterable(map(dominated_elements, f.monoms()) for f in fs))
    O = set([])
    for monom in get_support_basis(fs):
        O.update(dominated_elements(list(monom)))
    return sorted(O, key=grevlex, reverse=True)

def test_get_order_basis():
    _, x, y = ring('x,y', RR, grevlex)
    f = x**2 + x*y + y
    g = x**2 + x*y + x
    b = get_order_basis([f,g])
    assert b == [(2,0), (1,1), (1,0), (0,1), (0,0)]

def vector_representation(L, f):
    """
    Construct a matrix with terms in L
    """
    M = zeros((1, len(L)))
    for monom, coeff in f.terms():
        M[0, L.index(monom)] = coeff
    return M

def matrix_representation(L, I):
    """
    Construct a matrix with terms in L
    """
    M = zeros((len(I), len(L)))
    for i, f in enumerate(I):
        for monom, coeff in f.terms():
            M[i, L.index(monom)] = coeff
    return M

def unitary_basis(L, I):
    """
    Construct a matrix with terms in L
    """
    return row_normalize(matrix_representation(L, I))

def approx_unitary_basis(L, I, tau = 1e-4):
    """
    Construct a matrix with terms in L
    """
    M = rref(matrix_representation(L,I), tau)
    return M

    # Using QR instead of RREF because...
    #_, V = qr(matrix_representation(L, I))
    #V[abs(V) < eps] = 0
    #return row_normalize(V)

def test_approx_unitary_basis():
    R, I, _ = example_simple()
    x, y = R.symbols
    I_ = [x**3 - x,
          y**3 - y,
          x * y**2 + x **2 - 0.5 * y**2 - x - 0.5 * y,
          x**2 + x * y - 0.5 * y**2 - x - 0.5 * y,
          x**2 * y - 0.5 * y**2 - 0.5 * y,
         ]
    I_ = sorted([R(i) for i in I_], key=lambda t: grevlex(t.LM), reverse=True)

    L = get_support_basis(I)
    V = approx_unitary_basis(L, I, 0.001)

    V_ = row_normalize(matrix_representation(L, I_))
    print V
    print V_

def expand_order_ideal(L, B, W):
    """
    Expand the order ideal until it contains B
    """

    # Get all terms of polynomials whose leading terms are in L, but
    # other terms are not
    W_ = set(chain.from_iterable([B[i] for i in w.nonzero()[0]]
        for w in W if B[lt(w)[0]] in L)).difference(L)
    if len(W_) == 0:
        return L
    else:
        for m in W_:
            L.update(dominated_elements(m))
        return expand_order_ideal(L, B, W) 

def prune_columns(B, W):
    # Prune 0 columns
    W[abs(W) < eps] = 0
    keep = [i for (i, w) in enumerate(W.T) if norm(w) > eps]
    return sel(B, keep), W[:,keep]

def restrict_lt(L, B, W, order = grevlex):
    """
    Restrict the f in W to have leading terms in L
    """

    L = set(L)
    L = expand_order_ideal(L, B, W)
    W = array([w for w in W if B[lt(w)[0]] in L])
    return sorted(L, key=order, reverse=True), B, W

def test_restrict_lt1():
    _, x, y = ring('x,y', RR, grevlex)
    I = [x + y + 1, x**2 + y + 1]
    L = get_order_basis(I)
    B = get_support_basis(I)
    W = unitary_basis(B, I)

    L_, B_, W_ = restrict_lt(L, B, W)

    assert L_ == L
    assert np.allclose(W_, W)

def test_restrict_lt2():
    _, x, y = ring('x,y', RR, grevlex)
    I = [x*y**2, x**2 * y + x * y**2, x**3 * y**2]
    L = get_order_basis([x**2*y])
    B = get_support_basis(I)
    W = unitary_basis(B, I)

    L_, B_, W_ = restrict_lt(L, B, W)

    assert len(difference(L_, L)) > 0 and len(difference(L, L_)) == 0
    assert len(B_) == 2
    assert np.allclose(W_, W[:2,1:])

def approx_basis_extension(R, B, V, tau = 1e-4):
    """
    Extend the vector space $V$ with elements from $R$
    """
    b = to_syms(R, *B)
    A = [R(f) for f in V.dot(b)]
    B =  sum(([R(x) * f for f in A] for x in R.symbols), [])
    B_ = get_support_basis(A + B)

    V_A = approx_unitary_basis(B_, A, tau)
    V_B = approx_unitary_basis(B_, B, tau)

    # Zero out terms in in B that have the leading terms of A
    for va in V_A:
        idx, val = lt(va)
        for vb in V_B:
            vb -= va * vb[idx] / val

    # Remove approximately zero rows
    V_B = array([vb for vb in V_B if norm(vb) > tau])
    
    if V_B.shape[0] > 0:
        # Compute the e-truncated SVD and thus the ONB row space.
        _, _, V_B = truncated_svd(V_B, tau)
        V_B = rref(V_B, tau) # TODO: Set tau appropriately.
        #_, V_B = qr(V_B)
        #V_B[abs(V_B) < eps] = 0
    
    return B_, V_A, V_B

def test_approx_basis_extension():
    R, I, _ = example_simple()
    B = get_support_basis(I)
    V = approx_unitary_basis(B, I, 0.001)

    B, V, W = approx_basis_extension(R, B, V)

    print_space(R, B, lt_normalize(V))
    print "--"
    print_space(R, B, lt_normalize(W))

def border(R, O, order=grevlex):
    """
    Compute the border of O
    """
    dO = set([])
    for o in O: # Contains only monomials
        for i in xrange(len(R.symbols)):
            o_ = o[:i] + (o[i]+1,) + o[i+1:]
            if o_ not in O: dO.add(o_)

    return sorted(dO, key=grevlex, reverse=True)

def test_border():
    R, x, y = ring('x,y', RR, grevlex)
    O = get_order_basis([x, y])
    dO = border(R, O)
    assert dO == [(2,0), (1,1), (0,2)]

def fnnz(v):
    return v.nonzero()[0][0]

def compute_tau(V):
    r, s = V.shape
    c = max((max(v)/v[fnnz(v)] for v in abs(V)))

    tau = 1./np.sqrt(r + (s - r) * r**2 * c **2)
    print "tau", tau
    assert tau > eps

    return tau


def extend_basis(R, L, B, V, delta = 1e-3):
    """
    Keep extending the basis until you reach a fix point
    """
    # Compute approximate basis extension.
    tau = min(delta, compute_tau(V))
    B, V, W = approx_basis_extension(R, B, V, tau)
    assert not problem_kids(V)
    assert not problem_kids(W)

    # Restrict W_ to be in the order ideal $L$
    L, B, W = restrict_lt(L, B, W)
    assert not problem_kids(W)
    if len(W) > 0:
        # Combine the indices of B_
        B, V = prune_columns(B, np.vstack((V,W)))
        return extend_basis(R, L, B, V, delta)
    else:
        B, V = prune_columns(B, V)
        return L, B, V

def final_reduction(R, L, B, V, order = grevlex):
    """
    Final reduction algorithm
    Ensures that the terms have exactly one term in dO.
    """

    Lt = lti(B, V)
    O = L.difference(Lt)
    assert len(O) > 0

    VR = []

    # The rows are sorted in order of their term ordering
    # Dude, this is just reduced row form!
    for v in reversed(V):
        H = [t for t in v.nonzero()[0][1:] if B[t] not in O]
        for t in H:
            # Find the term in V_R with lt h, and remove it.
            w = filter(lambda w_ : lm(w_) == t, VR)[0]
            v -= w * v[t] / w[t]
        VR.append(v / lc(v))

    # Now return the indicator variables in the border set
    dO = [B.index(t) for t in border(R, O)]
    VB = array([v for v in VR if lt(v)[0] in dO])
    B, VB = prune_columns(B, VB)
    Gr = [R(f) for f in VB.dot(to_syms(R, *B))]
    Gr = sorted(Gr, key = lambda f: order(f.LM), reverse = True)

    O = sorted(O, key = order, reverse=True)
    return O, Gr

def compute(R, I, delta):
    """
    Compute border basis for I = (f1, ..., fr) that is delta close.
    Uses IABBA algorithm (Corr 4.11)
    """

    # L is order ideal spanned by supp(f_i).
    L = get_order_basis(I)
    B = get_support_basis(I)
    # Rows of V' are an approximate unitary basis {f1', ... , fr'}
    V = approx_unitary_basis(B, I, delta)
    assert not problem_kids(V)


    if do_debug:
        ipdb.set_trace()

    round = 0
    print round, V
    while True:
        L, B, V = extend_basis(R, L, B, V, delta)
        B, V = prune_columns(B, V)
        assert not problem_kids(V)
        print round, V
        print len(B)
        if len(B) > 1000:
            raise Exception("Ahhhh!")

        # Update order ideal $O$
        L = set(L)
        O = L.difference(lti(B, V))
        if len(difference(border(R,O), L)) > 0:
            # Expand order ideal $L$ with extension until border of $O$ matches.
            L.update(border(R, L))
        else:
            break
        round += 1
    # Apply final reduction algorithm.
    return final_reduction(R, L, B, V)

def test_compute():
    R, I, G = example_simple()
    O, G_ = compute(R, I, 0.001)
    B = sorted(border(R, O) + O, key = grevlex, reverse = True)
    Gm = matrix_representation(B, G)
    Gm_ = matrix_representation(B, G_)
    assert np.allclose(Gm, Gm_)

def test_compute_20():
    R, I, G = example_20()
    O, G_ = compute(R, I, 0.001)
    B = sorted(border(R, O) + O, key = grevlex, reverse = True)
    Gm = matrix_representation(B, G)
    Gm_ = matrix_representation(B, G_)
    assert np.allclose(Gm, Gm_)

def border_basis_divide(R, G, f):
    """
    Compute the mod remainder within the border basis
    """

    dO = [g.LM for g in G]
    # Get the leading term of f
    # Find the closest term in dO
    try :
        _, idx = min((sum(tuple_diff(f.LM, g)), i) 
                for i, g in enumerate(dO) 
                if min(tuple_diff(f.LM, g)) >= 0)
        g = G[idx]
        t, = to_syms(R, tuple_diff(f.LM, g.LM))
        f -= g * t * f.LC / g.LC
        return border_basis_divide(R, G, f)
    except ValueError:
        return f

def test_border_basis_divide():
    R, x, y = ring('x,y', RR, grevlex)
    G = [x**2 + x + 1, x *y + y, y**2 + x + 1]
    f = x**3 * y**2 - x * y**2 + x**2 + 2
    r = border_basis_divide(R, G, f)
    assert r == -3.0 * x - 1.0

def find_basis(G, t):
    """
    Find the term in the basis with leading term t
    """
    for g in G:
        if g.LM == t: 
            return g
    else:
        raise Exception("No such term")

def get_multiplication_matrices(R, O, G):
    """
    Let t1, ..., tn be the elements of O.
    Return the matrices M_1, ..., M_n
    """

    syms = [R(sym) for sym in R.symbols]
    dO = border(R, O)

    Ms = []
    for i in xrange(len(syms)):
        M = zeros((len(O), len(O)))
        for j, t in enumerate(O):
            t_ = tuple_incr(t, i) 
            if t_ in dO:
                b = find_basis(G,t_)
                for t_, c in b.terms()[1:]:
                    M[j, O.index(t_)] = c
            else:
                M[j, O.index(t_)] = 1
        Ms.append(M)

    return Ms

