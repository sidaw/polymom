#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Computational universes
"""

import numpy as np
import scipy as sc
import sympy as sp
from numpy import array, zeros, atleast_2d, diag
from numpy.linalg import svd, qr
import scipy.sparse
from scipy.sparse import csr_matrix, lil_matrix, vstack

from sympy import ring, RR, lex, grlex, grevlex, pprint
from util import *
from itertools import chain

import ipdb

eps = 1e-10

class ComputationalUniverse(object):
    """
    Represents a computational universe.
    """

    def __init__(self, R, tau=eps):
        self._ring = R
        self._symbols = [R(x) for x in R.symbols]
        self._nsymbols = len(self._symbols)
        self._order = R.order
        self._tau = tau

    @property
    def ring(self):
        """
        Get the ring
        """
        return self._ring

    @property
    def symbols(self):
        """
        Get the symbols of this computational universe
        """
        return self._symbols

    @property
    def nsymbols(self):
        """
        Get the number of symbols of this computational universe
        """
        return self._nsymbols

    @property
    def max_degrees(self):
        """
        Get the maximum degree of the symbol in the universe
        """
        raise NotImplementedError()

    @property
    def nterms(self):
        """
        How big is this universe?
        """
        raise NotImplementedError()

    def index(self, term):
        """
        Get numeric index for the term.
        Note that the convention is that the "largest term gets index
        0".
        e.g. (d,d,d) -> 0
        """
        raise NotImplementedError()

    def term(self, idx):
        """
        Get term for the numeric index
        Note that the convention is that the "largest term gets index
        0".
        e.g. 0 -> (d,d,d)
        """
        raise NotImplementedError()

    def monom(self, term):
        """
        Return the monomial from the ring representing this term tuple
        """
        return prod(x**i for x, i in zip(self._symbols, term))

    def max_degree(self, sym):
        """
        Get the maximum degree of the symbol in the universe
        """
        return self.max_degrees[self._symbols.index(sym)]

    def contains(self, f):
        """
        Does the computational universe contain elements in f?
        """
        raise NotImplementedError()

    def extend(self):
        """
        Extend by union with extensions in every dimension
        """
        raise NotImplementedError()

    def border(self, ts):
        """
        Return the border of elements
        """
        return BorderBasedUniverse.upper_border(chain.from_iterable(
            tuple_border(t) for t in ts), order=self._order)

    def as_vector(self, f):
        r"""
        Represent f as a vector in the universe.
        """
        raise NotImplementedError()

    def vector_space(self, fs):
        r"""
        Find a vector space spanning the set of polynomials f.
        """
        V = array([self.as_vector(f) for f in fs])
        _, V_ = srref(V, self._tau)
        return V_

    def as_poly(self, v):
        r"""
        Represent v as a polynomial in the universe.
        """
        raise NotImplementedError()

    def as_polys(self, V):
        r"""
        Represent v as a polynomial.
        """
        return [self.as_poly(v) for v in V]

class BorderBasedUniverse(ComputationalUniverse):
    """
    Represents a universe by its border
    """

    def __init__(self, R, border, tau=eps):
        super(BorderBasedUniverse, self).__init__(R, tau)
        self._border = border
        self._max_degrees = reduce(tuple_max, border)
        self._terms = self.__build_index()
        self._nterms = len(self._terms)

    def __build_index(self):
        """
        Build an index of terms
        """
        return sorted(
                set(chain.from_iterable(dominated_elements(b)
                    for b in self._border)),
                key=self._order, reverse=True)

    def as_vector(self, f):
        cols = [self.index(t) for t in f.monoms()]
        rows = [0 for _ in cols]
        nrows, ncols = 1, self._nterms
        data = [float(v) for v in f.values()]
        return csr_matrix((data, (rows, cols)), shape=(nrows, ncols))

    def as_poly(self, v):
        if isinstance(v, np.ndarray):
            cols, = v.nonzero()
            idxs = cols
        elif isinstance(v, sc.sparse.base.spmatrix):
            rows, cols = v.nonzero()
            idxs = zip(rows, cols)
        else:
            raise Exception("Invalid format")
        return sum(v[idx] * self.monom(self.term(i)) for (idx, i) in zip(idxs,cols))

    def vector_space(self, fs):
        r"""
        Find a vector space spanning the set of polynomials f.
        """
        data, rows, cols = [], [], []
        for i, f in enumerate(fs):
            cols.extend(self.index(t) for t in f.monoms())
            rows.extend(i for _ in f.monoms())
            data.extend(float(v) for v in f.values())
        nrows, ncols = len(fs), self._nterms
        return csr_matrix((data, (rows, cols)), shape=(nrows, ncols))

    def index(self, term):
        """
        Get numeric index for the term
        """
        return self._terms.index(term)

    def term(self, idx):
        """
        Get numeric index for the term
        """
        return self._terms[idx]

    def max_degrees(self):
        return self._max_degrees

    def contains(self, f):
        # assert type(f) == PolyElement
        if isinstance(f, sp.polys.rings.PolyElement):
            for t, _ in f.terms():
                if not self.contains(t):
                    return False
            return True
        elif isinstance(f, tuple):
            return BorderBasedUniverse.border_contains(self._border, f)

    @staticmethod
    def border_contains(L, t):
        """
        Does the border contain this term?
        """
        for b in L:
            if tuple_subs(b, t):
                return True
        return False

    @staticmethod
    def test_border_contains():
        """
        Test border contains
        """
        L = [(2, 1), (1, 2)]
        assert BorderBasedUniverse.border_contains(L, (1, 2))
        assert BorderBasedUniverse.border_contains(L, (1, 1))
        assert BorderBasedUniverse.border_contains(L, (0, 1))
        assert not BorderBasedUniverse.border_contains(L, (2, 2))

    @staticmethod
    def upper_border(L, order=grevlex):
        """
        Simplify a collection of elements so that it just contains the
        border.
        """
        L = set(L)
        # Remove something from the set iff it's border is already in
        # the set.
        for t in sorted(L, key=grevlex):
            if L.issuperset(tuple_border(t)):
                L.discard(t)
        return sorted(L, key=order, reverse=True)

    @staticmethod
    def test_upper_border():
        """
        Test simplify border
        """
        L = [(2, 1), (1, 1), (0, 1), (1, 2)]
        L_ = BorderBasedUniverse.upper_border(L, grevlex)
        assert L_ == [(2, 1), (1, 2), (0, 1)]

        L = [(0, 1), (1, 1), (2, 1), (1, 2)]
        L_ = BorderBasedUniverse.upper_border(L, grevlex)
        assert L_ == [(2, 1), (1, 2), (0, 1)]

        L = [(2, 1), (1, 2), (2, 0), (0, 2),]
        L_ = BorderBasedUniverse.upper_border(L, grevlex)
        assert L_ == L

    def extend(self, *Vs):
        """
        Extend the border by one
        Appropriately updates the indices of each of the matrices V
        """
        border = BorderBasedUniverse.upper_border(self._border +
                list(chain.from_iterable([tuple_incr(b, x) for b in self._border]
                    for x in xrange(self._nsymbols))))
        L = BorderBasedUniverse(self._ring, border)
        Vs = tuple(BorderBasedUniverse.update_vector(self, L, V) for V in Vs)

        return (L,) + Vs

    @staticmethod
    def test_extend():
        """
        Test extension
        """
        R = ring('x,y', RR)[0]
        L = BorderBasedUniverse(R, [(2, 1), (1, 2)])
        L.extend()
        assert L._border == [(3, 1), (2, 2), (1, 3)]

    @staticmethod
    def update_vector(old_universe, new_universe, arr):
        """
        Update arr to be in the new universe
        """
        nrows, _ = arr.shape
        ncols_ = new_universe.nterms()
        rows, cols = arr.nonzero()
        cols_ = [new_universe.index(old_universe.term(i)) for i in cols]
        return csr_matrix((arr.data, (rows, cols_)), shape=(nrows, ncols_))

    def extend_within(self, V, tau=None):
        r"""
        Extend the basis $V$ within the universe.
        Essentially, this computes $V⁺ \cap L$
        """
        if tau is None: tau = self._tau

        _, ncols = V.shape

        Wr, Wc = [], []
        data = []
        row_index = 0
        for i in xrange(self._nsymbols):
            for v in V:
                if isinstance(V, np.ndarray):
                    cols, = v.nonzero()
                elif isinstance(V, sc.sparse.base.spmatrix):
                    _, cols = v.nonzero()
                try:
                    cols = list(map(self.index,
                        (tuple_incr(t, i) for t in map(self.term, cols))))

                    # Remove any leading terms present in V (to ensure
                    # pairwise leading terms)
                    rows = zeros(len(cols))
                    w = csr_matrix((v.data, (rows, cols)), shape=(1, ncols))
                    for v in V:
                        idx, val = lt(v, tau)
                        if w[0, idx] != 0:
                            w = w - v * w[0, idx] / val

                    if norm(w) < self._tau:
                        continue

                    data.extend(w.data)
                    _, cols = w.nonzero()
                    Wc.extend(cols)
                    Wr.extend(row_index for _ in cols)
                    row_index += 1
                except ValueError:
                    # This term is outside our universe, ignore
                    pass
        nrows = row_index

        W = csr_matrix((data, (Wr, Wc)), shape=(nrows, ncols))
        ipdb.set_trace()
        if nrows > 0:
            # Compute the truncated svd
            # Stupid thing can't be done on sparse matrices.
            # Grumble
            _, _, W = truncated_svd(W.todense(), self._tau)
            _, W = srref(W, self._tau)
            W = csr_matrix(W)

        return W

    def stable_extension(self, V, tau=None):
        r"""
        Extend the basis $V$ within the universe till fix point.
        """
        if tau is None: tau = self._tau

        W = self.extend_within(V, tau)
        if W.shape[0] == 0:
            return V
        else:
            return self.stable_extension(vstack((V, W), 'csr'), tau)

    def supplementary_space(self, V):
        r"""
        Find a the supplementary space of V, such that L = B ⊕ V
        The supplementary space is the space.
        """
        dO = set([self.term(lm(v, self._tau)) for v in V])
        # Get everything less than dO
        O = set(chain.from_iterable(dominated_elements(o) for o in dO))
        O.difference_update(dO)
        return sorted(O, key=self._order, reverse=True)

    def contains_extension(self, B):
        r"""
        Let $B ⊆ L$.
        This function returns whether or not $B⁺ ⊆ L$.
        """
        for t in B:
            for i in xrange(self._nsymbols):
                if not self.contains(tuple_incr(t, i)):
                    return False
        return True

    @staticmethod
    def from_support(R, I, tau=eps):
        """
        Creates a border basis from the support of a set of polynomials I
        """
        border = BorderBasedUniverse.upper_border(
                chain.from_iterable(i.monoms() for i in I))
        return BorderBasedUniverse(R, border, tau)

class DegreeBoundedUniverse(BorderBasedUniverse):
    """
    Represents a universe by its border
    """

    def __init__(self, R, max_degree, tau=eps):
        border = [tuple(max_degree for _ in R.symbols)]
        super(DegreeBoundedUniverse, self).__init__(R, border, tau)
        self._max_degree = max_degree

    def index(self, term):
        """
        Get numeric index for the term.
        Note that the convention is that the "largest term gets index
        0".
        e.g. (d,d,d) -> 0
        """
        d, n = self._max_degree, self._nsymbols

        if self._order == lex:
            d_ = d+1
            # The index is simply max degree multiples.
            idx = 0
            for i, t in enumerate(reversed(term)):
                idx += (d_**i) * t
            return d_**n - 1 - idx
        else:
            return super(DegreeBoundedUniverse, self).index(term)

    def term(self, idx):
        """
        Get term for the numeric index
        Note that the convention is that the "largest term gets index
        0".
        e.g. 0 -> (d,d,d)
        """
        d, n = self._max_degree, self._nsymbols

        if self._order == lex:
            d_ = d+1
            idx = d_**n - idx - 1
            # The index is simply max degree multiples.
            term = [0 for _ in self._symbols]
            for i in xrange(self._nsymbols-1, -1, -1):
                term[i], idx = idx % d_, idx // d_
            return tuple(term)
        else:
            return super(DegreeBoundedUniverse, self).term(idx)

    @staticmethod
    def from_support(R, I, tau=eps):
        """
        Get the largest degree of terms in I
        """
        max_degree = max(max(map(max, i.monoms())) for i in I)
        return DegreeBoundedUniverse(R, max_degree, tau)

