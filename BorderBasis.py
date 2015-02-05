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
from util import to_syms, tuple_diff, tuple_incr, support, order_ideal, to_syms, row_normalize
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
        span = MonomialSpan.from_polynomials(R, fs, self.order)
        approximate_unitary(span, self.delta)

        while True:
            # Extend basis
            # Prune columns
            # Are the leading terms contained in the border?
            pass

        # Final reduction
        pass

