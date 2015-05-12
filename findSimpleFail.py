#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
test utilites for the MomentMatrix class, univariate skip moments, MoG,
and K by D mixtures.
"""

from cvxopt import matrix, sparse, solvers
import sympy as sp
import scipy as sc
import numpy as np
import mompy as mp
import ipdb
solvers.options['maxiters'] = 300
solvers.options['show_progress'] = True
solvers.options['feastol'] = 1e-7
solvers.options['abstol'] = 1e-5
solvers.options['reltol'] = 1e-7

def test_unimixture():
    print 'testing simple unimixture with a skipped observation'
    x = sp.symbols('x')
    
    constrs = [x+x**2-2, x+2*x**4-3, 2*x**2+3*x**4-5] # always solvable
    constrs = [x+x**2-1, x**2+2*x**3-1, 2*x**3+3*x**4-3, 2*x+x**3 - 0] # always solvable
    constrs = [x-1.5, x**2-2.5, x**5-16.5] # solvable with deg=3, but not deg=2
    constrs = [x-1.5, x**2-2.5, x**4-8.5] # solvable with deg=3, but not deg=2
    
    M = mp.MomentMatrix(3, [x], morder='grevlex')
    
    cin = mp.solvers.get_cvxopt_inputs(M, constrs)
    sol = solvers.sdp(matrix(sc.rand(*cin['c'].size)), Gs=cin['G'], \
                  hs=cin['h'], A=cin['A'], b=cin['b'])

    M.pretty_print(sol)

    print mp.extractors.extract_solutions_lasserre(M, sol['x'], Kmax = 2)
    return M,sol

def test_bimixture():
    print 'testing simple unimixture with a skipped observation'
    x,y = sp.symbols('x,y')
    constrs = [y+0.5, x-0.5, x+y-0, x**2+y-0, x+y**2-1, x+x**2-1, x*y + y**2, x-4*y - 2.5, y**2*x -0.5] # solvable with deg=3, but not deg=2
        
    M = mp.MomentMatrix(2, (x,y), morder='grevlex')
    
    cin = mp.solvers.get_cvxopt_inputs(M, constrs)
    sol = solvers.sdp(matrix(sc.rand(*cin['c'].size)), Gs=cin['G'], \
                  hs=cin['h'], A=cin['A'], b=cin['b'])

    M.pretty_print(sol)

    print mp.extractors.extract_solutions_lasserre(M, sol['x'], Kmax = 2)
    return M,sol

if __name__ == '__main__':
    M,sol=test_unimixture()
    print M.numeric_instance(sol['x'])

    #test_bimixture()
    
    
