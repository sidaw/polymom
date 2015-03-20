#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Constructing border bases
"""

from sys import maxint
import numpy as np
import sympy as sp
import scipy as sc
from numpy import array, zeros, atleast_2d, hstack, diag
from numpy.linalg import norm, svd, qr
from scipy.sparse import csr_matrix

from sympy import ring, RR, lex, grevlex, pprint
from util import *
from itertools import chain

from ComputationalUniverse import BorderBasedUniverse, DegreeBoundedUniverse

import globals

import ipdb

class Basis(object):
    """
    A basis supports some basis functionalities of quotienting and division.
    """

    def __init__(self, L, generators):
        """
        @param L - computational universe
        @param generators - Generators of the basis
        """
        self.L = L 
        self._generators = generators

    @property
    def generator_basis(self):
        """
        Return generators basis
        """
        return self._generators

    @property
    def quotient_basis(self):
        """
        Return quotient basis
        """

        raise NotImplementedError()

    def mod(self, f):
        """
        Find f mod I.
        """

        raise NotImplementedError()

    def multiplication_matrix(self, f):
        """
        Return the characteristic multiplication matrix of f.
        """
        Q = self.quotient_basis()
        Qi = [q.monoms()[0] for q in Q]
        d = len(Q)
        V = np.zeros((d,d))
        for i, q in enumerate(Q):
            r = self.mod(f*q)
            for t, c in r.terms():
                j = Qi.index(t)
                V[i,j] = c
        return V    

    def formal_multiplication_matrices(self):
        """
        Construct the multiplication matrices for each symbol
        """
        return [self.multiplication_matrix(sym) for sym in self.L.symbols]

    def zeros(self):
        """
        Find the zeros of the ideal by the companion method.
        """
        Ms = self.formal_multiplication_matrices()
        # Simultaneously diagonalize
        L, _ = simultaneously_diagonalize(Ms)
        return zip(*L)

class BorderBasis(Basis):
    """
    A border basis.
    """

    def __init__(self, L, O, V):
        super(BorderBasis, self).__init__(L, L.as_polys(V))
        self.O = O
        self.dO = L.border(O)
        self.V = V
        self._quotient_basis = [L.monom(o) for o in O]
        self.Q = BorderBasedUniverse(L.ring, O)

    def find_basis_nearest_lt(self, t):
        """
        Find the basis element with the nearest leading term
        """
        # Find the index closest to t
        closest, dist = None, maxint
        for t_ in self.dO:
            d = tuple_diff(t, t_)
            if any(idx < 0 for idx in d): continue
            dist_ = sum(d)
            if dist_ < dist:
                closest, dist = t_, dist_
        assert closest is not None

        return self._generators[self.dO.index(closest)]

    def find_basis_with_lt(self, t):
        """
        Find the basis element with leading term
        """
        idx = self.dO.index(t)
        return self._generators[idx]

    def quotient_basis(self):
        return self._quotient_basis

    def mod(self, f):
        """
        Compute the mod
        """
        while True:
            if f.LM in self.O:
                return f
            else:
                # Find the nearest element on the border, and eliminate
                b = self.find_basis_nearest_lt(f.LM)
                f -= f.LC/b.LC * self.L.monom(tuple_diff(f.LM, b.LM)) * b

    def formal_multiplication_matrices(self):
        """
        With border bases, we know that the formal multiplication
        matrices essentially only invoke terms on the border, so we can
        compute this super efficiently.
        """
        d = len(self.Q)
        Vs = [np.zeros((d,d)) for _ in xrange(self.L.nsymbols)]
        for i, V in enumerate(Vs):
            for j, b in enumerate(self.O):
                t = tuple_incr(b, i)
                if t in self.O:
                    V[j, self.Q.index(t)] = 1
                else:
                    b_ = self.find_basis_with_lt(t)
                    for t, c in b_.terms()[1:]:
                        V[j, self.Q.index(t)] = -c
        return Vs

class BorderBasisFactory(object):
    """
    Creates a border basis
    """

    def __init__(self, delta = eps, order = grevlex):
        self.delta = delta
        self.order = order

    def __final_reduction(self, L, V, O): 
        """
        The final reduction algorithm ensures that each term in the
        border is uniquely associated with each term in basis V
        @params: L - computation universe
        @params: V - O pre-basis
        @params: O - order ideal
        """
        assert len(O) > 0
        dO = set(L.index(t) for t in L.border(O))

        # Ensure we are in rref.  
        _, V = srref(V, self.delta)
        # Pick only those vectors on the border
        V = array([v for v in V if lm(v) in dO])

        # set leading coefficients to be 1
        V = lt_normalize(V)
        return O, V

    def __compute_tau(self, V):
        r, s = V.shape
        c = max((v.max()/lc(v, self.delta) for v in abs(V) if norm(v) > eps))

        tau = 1./np.sqrt(r + (s - r) * r**2 * c **2)
        #print "tau", tau
        assert tau > eps

        return tau

    def __inner_loop(self, L, V):
        """
        Inner loop: Find the L-stable span of V. If B be the
        supplementary space, terminate if $B⁺ ⊆ L$, otherwise, recurse
        with $L⁺$.
        """
        #ipdb.set_trace()
        # Get the stable extension
        tau = min(self.delta, self.__compute_tau(V))
        V = L.stable_extension(V, tau)

        B = L.supplementary_space(V) 
        # Check if we've reached fixed point.
        if not L.contains_extension(B):
            # TODO: An optimization is to extend L by the terms in B.
            globals.info.add_stage()
            L, V = L.extend(V)
            globals.info.L = L
            return self.__inner_loop(L, V)
        else:
            return L, V, B

    def generate(self, R, I):
        """
        Return a border basis for fs.
        """

        #ipdb.set_trace()
        # Get the computation universe.
        L = DegreeBoundedUniverse.from_support(R, I, tau=self.delta)

        globals.info = globals.AlgorithmInformation(L, I)

        # Get a linear basis for I
        V = L.vector_space(I)
        _, V = srref(V, self.delta) # Start sparse!
        V = csr_matrix(V)
        L, V, O = self.__inner_loop(L, V)

        #for v in V:
        #    print L.as_poly(lt_normalize(v))
        

        # Final reduction
        O, V = self.__final_reduction(L, V, O)
        return BorderBasis(L, O, V)

