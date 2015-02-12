#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Computational universes
"""

import numpy as np
import sympy as sp
from numpy import array, zeros, atleast_2d, hstack, vstack, diag
from numpy.linalg import norm, svd, qr

from sympy import ring, RR, lex, grevlex, pprint
from util import *
from itertools import chain

import ipdb

class ComputationalUniverse(object):
    """
    Represents a computational universe.
    """

    def __init__(self, R):
        self._ring = R
        self._symbols = [R(x) for x in R.symbols]
        self._nsymbols = len(self._symbols)

    @property
    def symbols(self):
        """
        Get the symbols of this computational universe
        """
        return self._symbols

    @property
    def max_degrees(self):
        """
        Get the maximum degree of the symbol in the universe
        """
        raise NotImplementedError()

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

    def as_vector(self, f):
        r"""
        Represent f as a vector in the universe.
        """
        raise NotImplementedError()

    def vector_space(self, fs):
        r"""
        Find a vector space spanning the set of polynomials f.
        """
        V = array(self.as_vector(f) for f in fs)
        _, V_ = srref(V)
        return V_

class BorderBasedUniverse(ComputationalUniverse):
    """
    Represents a universe by its border
    """ 

    def __init__(self, R, border):
        super(BorderBasedUniverse, self).__init__(R)
        self.border = border
        self._max_degrees = reduce(tuple_max, border)

    def max_degrees(self):
        return self._max_degrees

    def contains(self, f):
        # assert type(f) == PolyElement
        for t, _ in f.terms():
            if not BorderBasedUniverse.border_contains(self.border, t):
                return False
        return True

    @staticmethod
    def border_contains(L, t):
        """
        Does the border contain this term?
        """
        for b in L:
            if tuple_subs(b, t): return True
        return False

    @staticmethod
    def test_border_contains():
        """
        Test border contains
        """
        L = [(2,1), (1,2)]
        assert BorderBasedUniverse.border_contains(L, (1,2))
        assert BorderBasedUniverse.border_contains(L, (1,1))
        assert BorderBasedUniverse.border_contains(L, (0,1))
        assert not BorderBasedUniverse.border_contains(L, (2,2))

    @staticmethod
    def simplify_border(L):
        """
        Simplify a collection of elements so that it just contains the
        border.
        """
        it = iter(L)

        # Maintain a list of elements on the border.
        L_ = [next(it)]

        for l in it:
            # If anything contains something that is strictly less than the
            # other element in the border, don't keep it.
            for l_ in iter(L_):
                # If this element (l) is subsumed by something in the new
                # border, ignore it.
                if tuple_subs(l_, l): break
                # If this element (l) subsumes something in the new
                # border, remove from the new border
                elif tuple_subs(l, l_): del L_[L_.index(l_)]
            else:
                # If you've come this far, nothing in the new border
                # subsumes you, so add.
                L_.append(l)
        return L_

    @staticmethod
    def test_simplify_border():
        """
        Test simplify border
        """
        L = [(2,1), (1,1), (0,1), (1,2)]
        L_ = sorted(BorderBasedUniverse.simplify_border(L))
        assert L_ == [(1,2), (2,1)]

        L = [(0,1), (1,1), (2,1), (1,2)]
        L_ = sorted(BorderBasedUniverse.simplify_border(L))
        assert L_ == [(1,2), (2,1)]


    def extend(self, *Vs):
        """
        Extend the border by one
        Appropriately updates the indices of each of the matrices V
        """
        border = BorderBasedUniverse.simplify_border(self.border +
                list(chain.from_iterable([tuple_incr(b, x) for b in self.border]
                    for x in xrange(self._nsymbols))))
        # TODO: Update Vs
        return (BorderBasedUniverse(self._ring, border),) + Vs

    @staticmethod
    def test_extend():
        """
        Test extension
        """
        R = ring('x,y', RR)[0]
        L = BorderBasedUniverse(R, [(2,1), (1,2)])
        L.extend()
        assert L.border == [(3,1), (2,2), (1,3)]

    def extend_within(self, V):
        r"""
        Extend the basis $V$ within the universe.
        Essentially, this computes $V^+ \cap L$
        """
        raise NotImplementedError()

    def stable_extension(self, V):
        r"""
        Extend the basis $V$ within the universe till fix point.
        """
        W = self.extend_within(V)
        if len(W) == 0:
            return V
        else:
            return self.stable_extension(vstack((V,W)))

        raise NotImplementedError()

    def supplementary_space(self, V):
        r"""
        Find a the supplementary space of V, such that L = B ⊕ V
        """
        raise NotImplementedError()

    def contains_extension(self, v):
        r"""
        Let v be a indicator vector of basis elements $B ⊆ L$. 
        This function returns whether or not $B⁺ ⊆ L$.
        """
        raise NotImplementedError()

    @staticmethod
    def from_support(I):
        """
        Creates a border basis from the support of a set of polynomials I
        """

        raise NotImplementedError()

