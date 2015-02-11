#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Constructing border bases
"""

import numpy as np
import sympy as sp
from numpy import array, zeros, atleast_2d, hstack, diag
from numpy.linalg import norm, svd, qr

from sympy import ring, RR, lex, grevlex, pprint
from util import *
from itertools import chain

from MonomialSpan import MonomialSpan, approximate_unitary


class Basis(object):
    """
    A basis supports some basis functionalities of quotienting and division.
    """

    def __init__(self, R, generators):
        self.R = R 
        self.__generators = generators

    @property
    def generator_basis(self):
        """
        Return generators basis
        """

        raise NotImplementedError()

    @property
    def quotient_basis(self):
        """
        Return quotient basis
        """

        raise NotImplementedError()

    def quotient(self, f):
        """
        Find the quotient of f mod I.
        """

        raise NotImplementedError()

    def multiplication_matrix(self, f):
        """
        Return the characteristic multiplication matrix of f.
        """
        raise NotImplementedError()

class ComputationalUniverse(object):
    """
    Represents a computational universe.
    """

    def __init__(self, symbols):
        self._symbols = symbols

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

class BorderBasedUniverse(ComputationalUniverse):
    """
    Represents a universe by its border
    """

    def __init__(self, symbols, border):
        super(BorderBasedUniverse, self).__init__(symbols)
        self.border = border
        self.__max_degrees = reduce(tuple_max, border)

    def max_degrees(self):
        return self.__max_degrees

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
            if max(tuple_diff(t, b)) > 0:
                return False
        return True

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
            if not BorderBasedUniverse.border_contains(l, L_):
                L_.append(l)
        return L_

    def extend(self):
        """
        Extend the border by one
        """
        self.border = BorderBasedUniverse.simplify_border(self.border +
                sum([x * b for b in self.border] for x in self.symbols))

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
        V, W = self.extend_within(V)
        if len(W) == 0:
            return V
        else:
            return self.stable_extension(np.hstack((V,W)))

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


class BorderBasis(Basis):
    """
    A border basis.
    """

    def find_basis_nearest_lt(self, t):
        """
        Find the basis element with the nearest leading term
        """
        raise NotImplementedError()


    def find_basis_with_lt(self, t):
        """
        Find the basis element with leading term
        """
        raise NotImplementedError()

    def quotient(self, f):
        raise NotImplementedError()

class BorderBasisFactory(object):
    """
    Creates a border basis
    """

    def __init__(self, delta = 1e-3, order = grevlex):
        self.delta = delta
        self.order = order

    def __final_reduction(self, L, V, B): 
        raise NotImplementedError()

    def __inner_loop(self, L, V):
        # Get the stable extension
        V = L.stable_extension(V)

        B = L.supplementary_space(V) 
        # Check if we've reached fixed point.
        if not L.contains_extension(B):
            # TODO: An optimization is to extend L by the terms in B.
            return self.__inner_loop(L.extend(), V)
        else:
            return L, V, B

    def generate(self, R, I):
        """
        Return a border basis for fs.
        """

        # Get the computation universe.
        L = BorderBasedUniverse.from_support(support(I))
        # Get a linear basis for I
        V = L.__vector_space(I)
        L, V, B = self.__inner_loop(L, V)

        # Final reduction
        B, G = self.__final_reduction(L, V, B)
        return B, G

