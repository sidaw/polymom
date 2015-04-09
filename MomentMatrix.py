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
from cvxopt import matrix, sparse

def monomial_filter(mono, filter='even', debug=False):
        if filter is 'even':
            if debug and not mono==1:
                print str(mono) + ':\t' + str(all([(i%2)==0 for i in mono.as_poly().degree_list()]))
            return 1 if mono==1 else int(all([i%2==0 for i in mono.as_poly().degree_list()]))

class MomentMatrix(object):
    """
    class to handle moment matrices and localizing matrices,
    and to produce the right outputs for the cvxopt SDP solver
    degree: max degree of the basic monomials corresponding to each row,
    so the highest degree monomial in the entire moment matrix would be twice that
    """
    def __init__(self, degree, variables, morder='grevlex'):
        self.degree = degree
        self.vars = variables
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
        # you can build a dict and do this faster, but no point since we solve SDP later
        for i,yi in enumerate(self.matrix_monos):
            Ai[i] = coefdict.get(yi,0)
        return Ai


    # if provided, constraints should be a list of sympy polynomials that should be 0.
    def get_cvxopt_inputs(self, constraints = None, sparsemat = True, filter = 'even'):
        # many options for what c might be
        if filter is None:
            c = matrix(np.ones((self.num_matrix_monos, 1)))
        else:
            c = matrix([monomial_filter(yi, filter='even') for yi in self.matrix_monos], tc='d')
            
        num_constrs = len(constraints) if constraints is not None else 0
        
        Anp = np.zeros((num_constrs+1, self.num_matrix_monos))
        bnp = np.zeros((num_constrs+1,1))
        if constraints is not None:
            for i,constr in enumerate(constraints):
                Anp[i,:] = self.get_rowofA(constr)
        
        Anp[-1,0] = 1
        bnp[-1] = 1
        b = matrix(bnp)
        
        if sparsemat:
            Gs = [sparse(self.get_all_indicator_lists(), tc='d')]
            A = sparse(matrix(Anp))
        else:
            Gs = [matrix(self.get_all_indicator_lists(), tc='d')]
            A = matrix(Anp)
            
        hs = [matrix(np.zeros((self.num_row_monos,self.num_row_monos)))]    
        
        return {'c':c, 'Gs':Gs, 'hs':hs, 'A':A, 'b':b}


