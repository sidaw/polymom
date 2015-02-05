#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Methods to manipulate a span of monomials.
"""

import sympy as sp
import numpy as np
from sympy import grevlex
from numpy import array, zeros, atleast_2d, hstack, diag
from numpy.linalg import norm, svd, qr
from util import to_syms, tuple_diff, tuple_incr, support, order_ideal, to_syms, row_normalize

class MonomialSpan(object):
    """
    Represents a set of polynomials in a monomial span
    """

    def __init__(self, ring, basis, elems):
        self.R = ring
        self.O = basis
        self.M = elems
        self.representers = to_syms(self.R, *self.O[:3])

    def __repr__(self):
        return "[MonomialSpan with %d polynomials in %s...]" % (self.M.shape[0],
                self.representers)

    def __call__(self, *args):
        """
        Get element with args
        """
        pass

    def __getitem__(self, key):
        pass

    def __setitem__(self, key, value):
        pass

    def __mul__(self, monom):
        """
        Multiplies the basis elements of this span by this monomial -
        updates both the basis and the elements so contained.
        """
        pass

    def __add__(self, other):
        """
        Extend both spans of this and other, and return [this; other] in
        the new basis.
        """
        pass

    @staticmethod
    def from_polynomials(ring, fs, order=grevlex):
        """
        - Get the order ideal for all the monomial terms in fs.
        - Create an ordered list of these monomials
        - Represent the polynomials in the basis
        """
        O = order_ideal(fs, order)
        M = zeros((len(fs), len(O)))
        for i, f in enumerate(fs):
            for term, coeff in f.terms():
                M[i, O.index(term)] = coeff
        return MonomialSpan(ring, O, M)

def rref(M, tol = 1e-5):
    """
    Compute the reduced row echelon form
    """
    # TODO: This is very very inefficient!
    R = sp.matrix2numpy(sp.Matrix(A).rref(iszerofunc = lambda v : abs(v) < tau)[0],
            dtype=np.double)
    return row_normalize(R)

def approximate_unitary(span, tol = 1e-5):
    """
    Construct an approximate unitary matrix
    """
    span.M = rref(span.M)

