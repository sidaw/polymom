#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
classes and helper moment matrices and localizing matrices,
which takes contraints as input produce the right outputs
for the cvxopt SDP solver

Sketchy: tested using sympy 0.7.6 (the default distribution did not work)
and cvxopt
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
    class to handle moment matrices and localizing matrices, and to
    produce the right outputs for the cvxopt SDP solver degree: max
    degree of the basic monomials corresponding to each row, so the
    highest degree monomial in the entire moment matrix would be twice
    the provided degree.
    """
    def __init__(self, degree, variables, morder='grevlex'):
        """
        @param degree - highest degree of the first row/column of the
        moment matrix
        @param variables - list of sympy symbols
        """
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

        # This list correspond to the actual variables in the sdp solver
        self.matrix_monos = sorted(mn.itermonomials(self.vars, 2*self.degree),\
                                   key=monomial_key(morder, self.vars[::-1]))

        self.num_matrix_monos = len(self.matrix_monos)

        self.expanded_monos = []
        for yi in self.row_monos:
            for yj in self.row_monos:
                self.expanded_monos.append(yi*yj)
    
    def get_indicator_list(self, yj):
        return [-int(yi==yj) for yi in self.expanded_monos]

    def get_all_indicator_lists(self):
        allconstraints = [];
        for yi in self.matrix_monos:
            allconstraints += [self.get_indicator_list(yi)]
        return allconstraints

    def get_rowofA(self, constr):
        """
        @param - constr is a polynomial constraint expressed as a sympy
        polynomial. constr is h_j in Lasserre's notation,
        and represents contraints on entries of the moment matrix.
        """
        Ai = np.zeros(self.num_matrix_monos)
        coefdict = constr.as_coefficients_dict();
        # you can build a dict and do this faster, but no point since we solve SDP later
        for i,yi in enumerate(self.matrix_monos):
            Ai[i] = coefdict.get(yi,0)
        return Ai

    def get_cvxopt_inputs(self, constraints = None, sparsemat = True, filter = 'even'):
        """
        if provided, constraints should be a list of sympy polynomials that should be 0.

        """
        
        # Many for what c might be, not yet determined really
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
            G = [sparse(self.get_all_indicator_lists(), tc='d')]
            A = sparse(matrix(Anp))
        else:
            G = [matrix(self.get_all_indicator_lists(), tc='d')]
            A = matrix(Anp)
            
        h = [matrix(np.zeros((self.num_row_monos,self.num_row_monos)))]    
        
        return {'c':c, 'G':G, 'h':h, 'A':A, 'b':b}


class LocalizingMatrix(object):
    '''
    poly_gs is a list of polynomials that multiplies termwise
    to give the localizing matrices
    This class depends on the moment matrix class and has exactly the
    same monomials as the base moment matrix. So the SDP variables
    still corresponds to matrix_monos
    '''

    def __init__(self, mm, poly_g, morder='grevlex'):
        """
        @params - mm is a MomentMatrix object
        @params - poly_g the localizing polynomial
        """
        self.mm = mm
        self.poly_g = poly_g
        self.deg_g = poly_g.as_poly().total_degree()
        #there is no point to a constant localization matrix,
        #and it will cause crash because how sympy handles 1
        assert(self.deg_g>0)         
        rawmonos = mn.itermonomials(self.mm.vars, self.mm.degree-self.deg_g);
        self.row_monos = sorted(rawmonos,\
                                 key=monomial_key(morder, mm.vars[::-1]))
        self.num_row_monos = len(self.row_monos)
        self.expanded_polys = [];
        for yi in self.row_monos:
            for yj in self.row_monos:
                self.expanded_polys.append(sp.expand(poly_g*yi*yj))
        
    def get_indicator(self, yi):
        """
        @param - polynomial here is called g in Lasserre's notation
        and defines the underlying set K some parallel with
        MomentMatrix.get_indicator_list. Except now expanded_monos becomes
        expanded_polys
        """
        return [-int(pi.as_coefficients_dict().get(yi, 0)) for pi in self.expanded_polys]

    def get_indicators_list(self):
        allconstraints = [];
        for yi in self.mm.matrix_monos:
            allconstraints += [self.get_indicator(yi)]
        return allconstraints

    def get_cvxopt_Gh(self, sparsemat = True):
        """
        get the G and h corresponding to this localizing matrix

        """
        
        if sparsemat:
            G = sparse(self.get_indicators_list(), tc='d')
        else:
            G = matrix(self.get_indicators_list(), tc='d')
            
        h = matrix(np.zeros((self.num_row_monos,self.num_row_monos)))
        
        return {'G':G, 'h':h}
