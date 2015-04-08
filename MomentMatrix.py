#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
class to handle moment matrices and localizing matrices,
and to produce the right outputs for the cvxopt SDP solver

Notes: tested using sympy 0.7.6 (not the default distribution?) and cvxopt
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
    def __init__(self, degree, vars, morder='grevlex'):
        self.degree = degree
        self.vars = vars
        self.num_vars = len(self.vars)

        # this object is a list of all monomials
        # in num_vars variables up to degree degree
        rawmonos = mn.itermonomials(self.vars, self.degree);

        # the reverse here is a bit random..., but has to be done
        self.row_monos = sorted(rawmonos,\
                                 key=monomial_key(morder, self.vars[::-1]))
        self.num_row_monos = mn.monomial_count(self.num_vars, self.degree)

        # alphas are the vectors of exponents, num_mono by num_vars in size
        # actually serves no purpose right now
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
    def get_rowofA(self, constr):
        Ai = np.zeros(self.num_matrix_monos)
        coefdict = constr.as_coefficients_dict();
        for i,yi in enumerate(self.matrix_monos):
            Ai[i] = coefdict.get(yi,0)
        return Ai

    # if provided, constraints should be a list of sympy polynomials that should be 0.
    def get_cvxopt_inputs(self, constraints = None):
        # many options for what c might be
        c = np.ones((self.num_matrix_monos, 1))
        G = self.get_all_indicator_lists()
        h = np.zeros((self.num_row_monos,self.num_row_monos))

        if constraints is not None:
            num_constrs = len(constraints)
        else:
            num_constrs = 0
            
        A = np.zeros((num_constrs+1, self.num_matrix_monos))
        b = np.zeros((num_constrs+1,1))
        if constraints is not None:
            for i,constr in enumerate(constraints):
                A[i,:] = self.get_rowofA(constr)
            
        A[-1,0] = 1
        b[-1] = 1
        return {'c':c, 'G':G, 'h':h, 'A':A, 'b':b}


x = sp.symbols('x')
M = MomentMatrix(3, [x], morder='grevlex')
constrs = [x-1.5, x**2-2.5, x**3-4.5]
cin = M.get_cvxopt_inputs(constrs)

from cvxopt import matrix, solvers
sol = solvers.sdp(matrix(cin['c'], tc='d'), Gs=[matrix(cin['G'], tc='d')], \
                  hs=[matrix(cin['h'], tc='d')], A=matrix(cin['A'], tc='d'), b=matrix(cin['b'], tc='d'))

print sol['x']
