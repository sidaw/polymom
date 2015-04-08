#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
class to handle moment matrices and localizing matrices,
and to produce the right outputs for the cvxopt SDP solver
"""
#from __future__ import division
import sympy as sp
import numpy as np

import sympy.polys.monomials as mn
from sympy.polys.orderings import monomial_key

class MomentMatrix(object):
    """
    class to handle moment matrices and localizing matrices,
    and to produce the right outputs for the cvxopt SDP solver
    degree: max degree of the basic monomials corresponding to each row,
    so the highest degree monomial in the entire moment matrix would be twice that
    """
    def __init__(self, degree, varnames, morder='grevlex'):
        self.degree = degree
        self.varnames = varnames # a compatible list of names 'x1:5,y1:3,z'
        self.vars = sp.symbols(varnames)
        self.num_vars = len(self.vars)

        # this object is a list of all monomials
        # in num_vars variables up to degree degree
        rawmonos = mn.itermonomials(self.vars, self.degree);

        # the reverse here is a bit random... 
        self.row_monos = sorted(rawmonos,\
                                 key=monomial_key(morder, self.vars[::-1]))
        self.num_row_monos = mn.monomial_count(self.num_vars, self.degree)

        # alphas are the vectors of exponents, num_mono by num_vars in size
        self.alphas = np.zeros((self.num_row_monos, self.num_vars), dtype=np.int)
        for r,mono in enumerate(self.row_monos):
            exponents = mono.as_powers_dict()
            for c,var in enumerate(self.vars):
                self.alphas[r, c] = exponents[var]
                
        # this list correspond to the actual variables in the sdp solver
        self.matrix_monos = sorted(mn.itermonomials(self.vars, 2*self.degree),\
                                   key=monomial_key(morder, self.vars[::-1]))

        self.num_matrix_monos = len(self.matrix_monos);

        self.expanded_monos = [];
        for yi in self.row_monos:
            for yj in self.row_monos:
                self.expanded_monos.append(yi*yj);

    def get_indicator_list(self, monomial):
        return [-int(yi==monomial) for yi in self.expanded_monos]

    def get_all_indicator_lists(self):
        allconstraints = [];
        for yi in self.matrix_monos:
            allconstraints += [self.get_indicator_list(yi)]
        return allconstraints

    # constr is a polynomial constraint expressed as a sympy polynomial
    def get_rowofA(constr):
        Ai = np.zeros(self.num_matrix_monos)
        for i,yi in enumerate(self.matrix_monos)
            A[i] = constr.coeff(yi)
        return Ai

    # if provided, constraints should be a list of sympy polynomials that should be 0.
    def get_cvxopt_inputs(self, constraints = None):
        # many options for what c might be
        c = np.ones(self.num_matrix_monos, 1)
        G = self.get_all_indicator_lists()
        h = np.zeros(self.num_matrix_monos)

        numconstrs = len(constraints)
        
