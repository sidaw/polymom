#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
test utilites for the MomentMatrix class, univariate skip moments, MoG,
and K by D mixtures.
"""

from cvxopt import matrix, sparse, solvers
import sympy as sp
import numpy as np
import MomentMatrix as mm

def test_unimixture():
    print 'testing simple unimixture with a skipped observation'
    x = sp.symbols('x')
    M = mm.MomentMatrix(3, [x], morder='grevlex')
    constrs = [x-1.5, x**2-2.5, x**4-8.5]
    cin = M.get_cvxopt_inputs(constrs)
    sol = solvers.sdp(cin['c'], Gs=cin['Gs'], \
                  hs=cin['hs'], A=cin['A'], b=cin['b'])

    print sol['x']
    print abs(sol['x'][3]-4.5)
    assert(abs(sol['x'][3]-4.5) <= 1e-5)
    return M,sol

# helper function to generate coeffs of the Gaussian moments
# they are non-neg and equal in abs to the coeffs hermite polynomials
def hermite_coeffs(N=6):
    K = N
    A = np.zeros((N,K), dtype=np.int)
    # the recurrence formula to get coefficients of the hermite polynomails
    A[0,0] = 1; A[1,1] = 1; #A[2,0]=-1; A[2,2]=1;
    for n in range(1,N-1):
        for k in range(K):
            A[n+1,k] = -n*A[n-1,k] if k==0 else A[n,k-1] - n*A[n-1,k]
    return A

def test_1dmog(mus=[-1., 4.], sigs=[1., 1.], pis=[0.5, 0.5], deg = 4):
    print 'testing 1d mixture of Gaussians'
    K = len(mus)
    mu,sig = sp.symbols('mu,sigma')
    M = mm.MomentMatrix(deg, [mu, sig], morder='grevlex')

    num_constrs = 9; # so observe num_constrs-1 moments
    H = abs(hermite_coeffs(num_constrs))
    constrs = [0]*num_constrs
    
    for order in range(num_constrs):
        for i in range(order+1):
            constrs[order] = constrs[order] + H[order,i]* mu**(i) * sig**(order-i)
        constrsval = 0
        for k in range(K):
            constrsval += pis[k]*constrs[order].evalf(subs={mu:mus[k], sig:sigs[k]})
        constrs[order] -= constrsval
    print constrs
        
    cin = M.get_cvxopt_inputs(constrs[1:])
    sol = solvers.sdp(cin['c'], Gs=cin['Gs'], \
                  hs=cin['hs'], A=cin['A'], b=cin['b'])
    
    print M.matrix_monos
    print sol['x']
    print cin['c']
    #import pdb; pdb.set_trace()
    return M,sol

# K: num components, D: dimensions, pis: the mixture coefficients
# deg is the highest degree of the row_monos, and degobs is the highest observed moment
def test_K_by_D(K=3, D=3, pis=[0.25, 0.25, 0.5], deg=1, degobs=3):
    print 'testing the K by D mixture'
    np.random.seed(0); params = np.random.randint(0,5,size=(K,D))
    print 'the true parameters (K by D): '+str(params) + '\n' + str(pis)
    xs = sp.symbols('x1:'+str(D+1))
    print xs
    M = mm.MomentMatrix(deg, xs, morder='grevlex')

    constrs = [0] * len(M.matrix_monos)
    for i,mono in enumerate(M.matrix_monos):
        constrs[i] = mono
        constrsval = 0;
        for k in range(K):
            subsk = {xs[i]:params[k,i] for i in range(D)};
            constrsval += pis[k]*mono.evalf(subs=subsk)
        constrs[i] -= constrsval
    print constrs
    filtered_constrs = [constr for constr in constrs[1:] if constr.as_poly().total_degree()<=degobs]
    print filtered_constrs
    cin = M.get_cvxopt_inputs(filtered_constrs)
    sol = solvers.sdp(cin['c'], Gs=cin['Gs'], \
                  hs=cin['hs'], A=cin['A'], b=cin['b'])

    print M.matrix_monos
    print sol['x']
    return M,sol

Muni,sol_uni=test_unimixture()
M_mog,sol_mog=test_1dmog()
M_KbyD,sol_KbyD=test_K_by_D(K=3,D=3,pis=[0.25,0.25,0.5], deg=2, degobs=3)
M_KbyD_underdet,sol_KbyD_underdet=test_K_by_D(K=7,D=4,pis=[0.1,0.1,0.1,0.2,0.2,0.2,0.1],deg=3,degobs=3)

