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
        pass


    def find_basis_with_lt(self, t):
        """
        Find the basis element with leading term
        """
        pass

    def quotient(self, f):
        pass

class BorderBasisFactory(object):
    """
    Creates a border basis
    """

    def __init__(self, delta = 1e-3, order = grevlex):
        self.delta = delta
        self.order = order

    def __final_reduction(self, B): 
        pass

    def generate(self, R, fs):
        """
        Return a border basis for fs.
        """

        # Store the computation universe


        span = MonomialSpan.from_polynomials(R, fs, self.order)
        approximate_unitary(span, self.delta)

        while True:
            # Extend basis
            # Prune columns
            # Are the leading terms contained in the border?
            pass

        # Final reduction
        pass

