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
from util import hermite_coeffs
import ipdb
solvers.options['maxiters'] = 100
solvers.options['show_progress'] = True

def test_unimixture():
    print 'testing simple unimixture with a skipped observation'
    x = sp.symbols('x')
    M = mm.MomentMatrix(3, [x], morder='grevlex')
    constrs = [x-1.5, x**2-2.5, x**4-8.5]
    cin = M.get_cvxopt_inputs(constrs)
    sol = solvers.sdp(cin['c'], Gs=cin['G'], \
                  hs=cin['h'], A=cin['A'], b=cin['b'])

    print sol['x']
    print abs(sol['x'][3]-4.5)
    assert(abs(sol['x'][3]-4.5) <= 1e-5)
    print M.extract_solutions_lasserre(sol['x'], Kmax = 2)
    return M,sol


def test_1dmog(mus=[-2., 2.], sigs=[1., np.sqrt(3.)], pis=[0.5, 0.5], deg = 3):
    print 'testing 1d mixture of Gaussians'
    # constraints on parameters here
    
    K = len(mus)
    mu,sig = sp.symbols('mu,sigma')
    M = mm.MomentMatrix(deg, [mu, sig], morder='grevlex')

    num_constrs = 7; # so observe num_constrs-1 moments
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

    gs = [sig-1, 30-mu**2, 5-sig**2]
    #gs = [sig-1]
    locmatrices = [mm.LocalizingMatrix(M, g) for g in gs]
    Ghs = [lm.get_cvxopt_Gh() for lm in locmatrices]

    Gs=cin['G'] + [Gh['G'] for Gh in Ghs]
    hs=cin['h'] + [Gh['h'] for Gh in Ghs]
    
    sol = solvers.sdp(cin['c'],Gs=Gs, \
                  hs=hs, A=cin['A'], b=cin['b'])

    for i,mono in enumerate(M.matrix_monos):
        trueval = 0;
        if i>0:
            for k in range(K):
                trueval += pis[k]*mono.evalf(subs={mu:mus[k], sig:sigs[k]})
        else:
            trueval = 1
        print '%s:\t%f\t%f' % (str(mono), sol['x'][i], trueval)
    print M.extract_solutions_lasserre(sol['x'], Kmax=len(mus))
    print mm.sep_alternating_solver(M, constrs, 2, maxiter=30000)
    return M,sol

# K: num components, D: dimensions, pis: the mixture coefficients
# deg is the highest degree of the row_monos, and degobs is the highest observed moment
def test_K_by_D(K=3, D=3, pis=[0.25, 0.25, 0.5], deg=1, degobs=3):
    print 'testing the K by D mixture'
    np.random.seed(0); params = np.random.randint(0,10,size=(K,D))
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
        
    #print constrs
    filtered_constrs = [constr for constr in constrs[1:] if constr.as_poly().total_degree()<=degobs]
    print filtered_constrs
    cin = M.get_cvxopt_inputs(filtered_constrs)
    sol = solvers.sdp(cin['c'], Gs=cin['G'], \
                  hs=cin['h'], A=cin['A'], b=cin['b'])

    print 'Mono:\tEstimmated\tTrue'
    for i,mono in enumerate(M.matrix_monos):
        trueval = 0;
        if i>0:
            for k in range(K):
                subsk = {xs[i]:params[k,i] for i in range(D)};
                trueval += pis[k]*mono.evalf(subs=subsk)
        else:
            trueval = 1
        print '%s:\t%f\t%f' % (str(mono), sol['x'][i], trueval)
    print M.extract_solutions_lasserre(sol['x'], Kmax=K)
    return M,sol

# K: num components, D: dimensions
# deg is the highest degree of the row_monos, and degobs is the highest observed moment
# I guess we should do naive ICA, sine the other version is already solved..
def test_ICA(K=3, D=3, deg=2, degobs=4):
    print 'testing the K by D HMM with generated parameters'
    np.random.seed(0); A = np.random.randn((D,K))
    print 'the true parameters (D by K): '+str(A) + '\n'
    As = sp.symbols('A1:'+str(D+1))
    print As
    M = mm.MomentMatrix(deg, As, morder='grevlex')

if __name__ == '__main__':
    M_mog,sol_mog=test_1dmog(mus=[-2., 2.], sigs=[1., np.sqrt(3.)], pis=[0.5, 0.5], deg = 3)
    
    # this is  equivalent to the ICA formulation in Anandkumar
    K_ica = 7
    M_ICA,sol_ICA=test_K_by_D(K=K_ica,D=5,pis=[1.0/K_ica]*K_ica,deg=3,degobs=4)

    ipdb.set_trace()
    
    Muni,sol_uni=test_unimixture()
    M_mog,sol_mog=test_1dmog(mus=[-2., 2.], sigs=[1., np.sqrt(3.)], pis=[0.5, 0.5], deg = 3)

    Kmog = 2;
    pis = np.random.rand(Kmog); pis = pis/sum(pis)
    mus = (-5+10*np.random.rand(2)).tolist()
    print mus
    M_mog,sol_mog=test_1dmog(mus=[5,-3], \
                              sigs=(1+np.random.rand(2)).tolist(), pis=pis, deg = 3)

    M_KbyD,sol_KbyD=test_K_by_D(K=3,D=3,pis=[0.25,0.25,0.5], deg=2, degobs=3)
    M_KbyD_underdet,sol_KbyD_underdet=test_K_by_D(K=7,D=5,pis=[0.1,0.1,0.1,0.2,0.2,0.2,0.1],deg=3,degobs=3)
    
    Ktry = 15
    pis = np.random.rand(Ktry); pis = pis/sum(pis);
    ## # change to D=4 for a failure
    M_KbyD_underdet,sol_KbyD_underdet=test_K_by_D(K=Ktry,D=5,pis=pis,deg=3,degobs=4)


