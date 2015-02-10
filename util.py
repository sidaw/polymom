#!/usr/bin/env python2.7
"""
Various utility methods
"""
import operator
from itertools import chain
from sympy import grevlex
from numpy import array, zeros
from numpy.linalg import norm
import ipdb

eps = 1e-16

def tuple_add(t1, t2):
    """Elementwise addition of two tuples"""
    return tuple( t1[i] + t2[i] for i in xrange(len(t1)) )

def tuple_diff(t1, t2):
    """Elementwise addition of two tuples"""
    return tuple( t1[i] - t2[i] for i in xrange(len(t1)) )

def tuple_min(t1, t2):
    return tuple(min(a, b) for a, b in zip(t1, t2))

def tuple_max(t1, t2):
    return tuple(max(a, b) for a, b in zip(t1, t2))

def tuple_incr(t1, idx, val=1):
    return t1[:idx] + (t1[idx]+val,) + t1[idx+1:]

def nonzeros(lst):
    """Return non-zero indices of a list"""
    return (i for i in xrange(len(lst)) if lst[i] > 0)

def first(iterable, default=None, key=None):
    if key is None:
        for el in iterable:
            return el
    else:
        for el in iterable:
            return el
    return default

def prod(iterable):
    return reduce(operator.mul, iterable, 1)

def to_syms(R, *monoms):
    """
    Get the symbols of an ideal I
    """
    return [prod(R(R.symbols[i])**j 
                for (i, j) in enumerate(monom)) 
                    for monom in monoms]

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


def support(fs, order=grevlex):
    """
    Get the terms spanned by support of
    f_1, ... f_n
    """
    O = set(chain.from_iterable(f.monoms() for f in fs))
    return sorted(O, key=grevlex, reverse=True)

def order_ideal(fs, order=grevlex):
    """
    Return the order ideal spanned by these monomials.
    """
    O = set([])
    for t in support(fs, order):
        if t not in O:
            O.update(dominated_elements(list(t)))
    return sorted(O, key=grevlex, reverse=True)

def lt(arr, tau = 0):
    """
    Get the leading term of arr > tau
    """
    for idx, elem in enumerate(arr):
        if abs(elem) > tau:
            return idx, elem
    return 0, arr[0]

def lm(arr):
    return lt(arr)[0]

def lc(arr):
    return lt(arr)[1]

def lt_normalize(R):
    """
    Normalize to have the max term be 1
    """
    rows, _ = R.shape
    for r in xrange(rows):
        R[r,:] /= max(abs(R[r,:]))
    return R

def row_normalize(R, tau = eps, ord=None):
    """
    Normalize rows to have unit norm
    """
    for r in R:
        li = norm(r, ord=ord)
        if li < tau:
            r[:] = 0
        else:
            r /= li
    return R

def row_reduce(R, tau = eps):
    """
    Zero all rows with leading term from below
    """
    nrows, _ = R.shape
    for i in xrange(nrows-1, 0, -1):
        k, _ = lt(R[i,:], tau)
        for j in xrange(i):
            R[j, :] -= R[i,:] * R[j,k] / R[i,k]

    return R

def srref(A, tau = eps):
    """
    Compute the stabilized row reduced echelon form.
    """
    m, n = A.shape

    Q = []
    R = zeros((min(m,n), n)) # Rows

    for i in xrange(n):
        # Remove any contribution from previous rows
        ai = array(A[:, i]) # Make a copy.
        for j, qj in enumerate(Q):
            R[j, i] = ai.dot(qj)
            ai -= ai.dot(qj) * qj
        
        li = norm(ai)
        if li > tau:
            assert len(Q) < min(m,n)
            # Add a new column to Q
            Q.append(ai / li)
            # And write down the contribution
            R[i, i] = li

    # Convert to reduced row echelon form
    row_reduce(R, tau)

    # row_normalize
    row_normalize(R, tau)

    return array(Q).T, R

