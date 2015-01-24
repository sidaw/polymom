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
from numpy import array, zeros, atleast_2d, hstack, diag
from numpy.linalg import norm, svd, qr

from sympy import ring, RR, lex, grevlex, pprint
from util import to_syms
from itertools import chain

import ipdb

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

def row_normalize(R):
    """
    Normalize rows to have unit norm
    """
    return array([r / norm(r) for r in R if norm(r) > 1e-10])

def lt_normalize(R):
    """
    Normalize to have the max term be 1
    """
    rows, _ = R.shape
    for r in xrange(rows):
        R[r,:] /= max(abs(R[r,:]))
    return R

def lt(arr, tau = 0):
    """
    Get the leading term of arr > tau
    """
    return next((idx, elem) for (idx, elem) in enumerate(arr) if abs(elem) > tau)

def lm(arr):
    return lt(arr)[0]

def lc(arr):
    return lt(arr)[1]

def coeff(B, v, term):
    return v[B.index(term)]

def lti(B, V):
    """
    Leading term ideal
    """
    return [B[lt(f)[0]] for f in V]

def rref(A, tau):
    """
    a1, ..., an are columns of $A$. 
    This routine computes a reduced row echelon form with tolerance tau.
    """

    R = sp.matrix2numpy(sp.Matrix(A).rref(iszerofunc = lambda v : abs(v) < tau)[0],
            dtype=np.double)
    return row_normalize(R)

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
    C_ = rref(B, tau)
    print C
    print C_
    assert max(abs(C - C_).flatten()) < tau


    B = [[0.0004, 0.6755, -0.5089, -0.5068, -0.1667], [0, -0.3812, -0.3735, -0.3812, 0.7548]]

def example_trivial():
    """
    My own example.
    """

    R, x, y = ring('x,y', RR, grevlex)

    I = [x**2 - y**2 - 1,
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

def dominated_elements(lst, idx = 0):
    """
    Iterates over all elements that are dominated by the input list.
    For example, (2,1) returns [(2,1), (2,0), (1,1), (1,0), (0,0), (0,1)]
    """

    # Stupid check
    if type(lst) != list: lst = list(lst)

    # Yield (a copy of) this element
    yield tuple(lst)

    # Update all subsequent indices
    for idx_ in xrange(idx, len(lst)):
        tmp = lst[idx]

        # Ticker down this index
        while lst[idx] > 0:
            lst[idx] -= 1
            for elem in dominated_elements(lst, idx+1): yield elem
        lst[idx] = tmp

def test_dominated_elements():
    lst = [(1,2), (2,1)]
    L = dominated_elements(lst)
    assert (0,0) in L
    assert (1,0) in L
    assert (0,1) in L
    assert (1,1) in L
    assert (2,0) in L
    assert (0,2) in L
    assert (1,2) in L
    assert (2,1) in L
    assert (2,2) not in L

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

def approx_unitary_basis(L, I, tau):
    """
    Construct a matrix with terms in L
    """
    # Using QR instead of RREF because...
    #return rref(matrix_representation(L,I), tau)

    _, V = qr(matrix_representation(L, I))
    V[abs(V) < 1e-10] = 0
    return row_normalize(V)

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

def truncated_svd(M, epsilon = 0.001):
    """
    Computed the truncated version of M from SVD
    """
    U, S, V = svd(M)
    S = S[S > epsilon]
    return U[:, :len(S)], S, V[:len(S),:]

def expand_order_ideal(L, B, W):
    """
    Expand the order ideal
    """

    # Get all terms whose leading terms are in L
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
    keep = [i for (i, w) in enumerate(W.T) if norm(w) > 1e-10]
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

def approx_basis_extension(R, B, V, tau = 0.001):
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
    
    # Compute the e-truncated SVD and thus the ONB row space.
    _, _, V_B = truncated_svd(V_B, tau)
    #V_B = rref(V_B, tau) # TODO: Set tau appropriately.
    _, V_B = qr(V_B)
    V_B[abs(V_B) < 1e-10] = 0
    
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

def extend_basis(R, L, B, V):
    """
    Keep extending the basis until you reach a fix point
    """
    # Compute approximate basis extension.
    B, V, W = approx_basis_extension(R, B, V)

    # Restrict W_ to be in the order ideal $L$
    L, B, W = restrict_lt(L, B, W)
    if len(W) > 0:
        # Combine the indices of B_
        B, V = prune_columns(B, np.vstack((V,W)))
        return extend_basis(R, L, B, V)
    else:
        return L, B, V

def final_reduction(R, L, B, V):
    """
    Final reduction algorithm
    Ensures that the terms have exactly one term in dO.
    """

    Lt = lti(B, V)
    O = L.difference(Lt)

    VR = []

    # The rows are sorted in order of their term ordering
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
    return prune_columns(B, VB)

def compute(R, I, delta):
    """
    Compute border basis for I = (f1, ..., fr) that is delta close.
    Uses IABBA algorithm (Corr 4.11)
    """

    # L is order ideal spanned by supp(f_i).
    L = get_order_basis(I)
    B = get_support_basis(I)
    # Rows of V' are an approximate unitary basis {f1', ... , fr'}
    V = approx_unitary_basis(B, I, 0.0001)

    while True:
        L, B, V = extend_basis(R, L, B, V)
        B, V = prune_columns(B, V)

        # Update order ideal $O$
        L = set(L)
        O = L.difference(lti(B, V))
        if len(difference(border(R,O), L)) > 0:
            # Expand order ideal $L$ with extension until border of $O$ matches.
            L.update(border(R, L))
        else:
            break
    # Apply final reduction algorithm.
    ipdb.set_trace()
    return final_reduction(R, L, B, V)

def test_compute():
    R, I, G = example_simple()
    B, G_ = compute(R, I, 0.001)
    G = matrix_representation(B, G)
    assert np.allclose(G, G_)


