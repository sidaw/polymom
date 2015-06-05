# coding: utf-8

# # Mixture of Gaussians

import util
from util import *
import numpy as np
import models
import models.GaussianMixtures
import mompy as mp
import cvxopt
import sys
from operator import mul
import sympy as sp
from collections import Counter
import models.GaussianMixtures
from itertools import *

from cvxopt import solvers
solvers.options['show_progress'] = True
DEGMM = 3
DEGOB = 3
SPHERICAL = False

def get_sumto1constraints(syms, maxdeg = 4):
    if maxdeg < 0: return []
    P = len(syms)
    sum1 = -1
    for sym in syms:
        sum1 = sum1 + sym
    sum1eqs = []
    for j in range(1,maxdeg):
        slices = combinations_with_replacement(range(P), j)
        for s in slices:
            currenteq = sum1
            for i in s:
                currenteq = currenteq * syms[i]
            sum1eqs.append(sp.expand(currenteq))
    return sum1eqs

def M_polygmm(gm, X, degmm=DEGMM, degobs=DEGOB):
    tol= 1e-2
    k = gm.k
    gm.polymom_init_symbols(spherical = SPHERICAL)
    monos = gm.polymom_monos(degmm)
    constraints = gm.polymom_all_constraints(degobs)
    
    xis = gm.sym_means
    covs = gm.sym_covs
    sym_all = xis + covs
    MM = mp.MomentMatrix(degmm, sym_all, morder='grevlex', monos=monos)
    constraints_noisy = gm.polymom_all_constraints_samples(degobs, X) + get_sumto1constraints(xis, degobs)
        
    solsdp_noisy = mp.solvers.solve_generalized_mom_coneqp(MM, constraints_noisy)
        
        
    #sol_noisy = mp.extractors.extract_solutions_dreesen_proto(MM, solsdp_noisy['x'], Kmax = k)
    sol_noisy = mp.extractors.extract_solutions_lasserre(MM, solsdp_noisy['x'], Kmax = k)
    # M should always be k by d
    Mlist = []
    Clist = []
    for dim in xis:
        Mlist.append(sol_noisy[dim])
    for dim in covs:
        Clist.append(sol_noisy[dim])
    
    M_ = sc.column_stack(Mlist)
    C_ = sc.column_stack(Clist)
    return M_,C_

def M_polymom(gm, X, degmm=DEGMM, degobs=DEGOB):
    tol= 1e-2
    k = gm.k
    gm.polymom_init_symbols(spherical = SPHERICAL)
    monos = gm.polymom_monos(degmm)
    constraints = gm.polymom_all_constraints(degobs)
    xis = gm.sym_means
    covs = gm.sym_covs
    sym_all = xis + covs
    MM = mp.MomentMatrix(degmm, sym_all, morder='grevlex', monos=monos)
    constraints_noisy = gm.polymom_all_constraints_samples(degobs, X) + get_sumto1constraints(xis, degobs)
    solsdp_noisy = mp.solvers.solve_basic_constraints(MM, constraints_noisy, slack = 1e-5)
        
    #sol_noisy = mp.extractors.extract_solutions_dreesen_proto(MM, solsdp_noisy['x'], Kmax = k)
    sol_noisy = mp.extractors.extract_solutions_lasserre(MM, solsdp_noisy['x'], Kmax = k)
    # M should always be k by d
    Mlist = []
    Clist = []
    for dim in xis:
        Mlist.append(sol_noisy[dim])
    for dim in covs:
        Clist.append(sol_noisy[dim])
    
    M_ = sc.column_stack(Mlist)
    C_ = sc.column_stack(Clist)
    return M_,C_

def M_polymomconvexiter(gm, X, degmm=3, degobs=5):
    tol= 1e-2
    k = gm.k
    
    monos = gm.polymom_monos(degmm)
    constraints = gm.polymom_all_constraints(degobs)
    xis = gm.sym_means
    covs = gm.sym_covs
    sym_all = xis + covs
    MM = mp.MomentMatrix(degmm, sym_all, morder='grevlex', monos=monos)
    constraints_noisy = gm.polymom_all_constraints_samples(degobs, X)

    solsdp_noisy = mp.solvers.solve_moments_with_convexiterations(MM, constraints_noisy, k, maxiter = 2);

    #sol_noisy = mp.extractors.extract_solutions_dreesen_proto(MM, solsdp_noisy['x'], Kmax = k)
    sol_noisy = mp.extractors.extract_solutions_lasserre(MM, solsdp_noisy['x'], Kmax = k)
    # M should always be k by d
    Mlist = []
    Clist = []
    for dim in xis:
        Mlist.append(sol_noisy[dim])
    for dim in covs:
        Clist.append(sol_noisy[dim])
    
    M_ = sc.column_stack(Mlist)
    C_ = sc.column_stack(Clist)
    return M_,C_

def M_Spectral(gm, X):
    fname = "gmm-3-10-0.7.npz"
    k = gm.k
    import algos.GaussianMixturesSpectral as gms
    M_ = gms.find_means(X, k).T
    assert(M_.shape == (gm.k, gm.d))
    return M_,None

def M_EM(gm, X):
    from sklearn import mixture
    k = gm.k
    sklgmm = mixture.GMM(n_components=k, covariance_type='diag', n_init=5, n_iter = 10, thresh = 1e-2)
    sklgmm.fit(X)
    return sklgmm.means_, sklgmm.covars_

def M_true(gm, X):
    M_ = sc.random.permutation(gm.means.T)
    Clist = []
    for covmat in gm.sigmas:
        Clist += [sc.diag(covmat).tolist()]
    C_ = sc.random.permutation(sc.row_stack(Clist))
    return M_, C_

def test_all_methods(args):
    k = args.k
    d = args.d
    numsamp = args.N
    typemean = args.typemean #'rotatedhypercube'
    typecov = args.typecov
    numtrials = args.trials
    sc.random.seed(args.seed)
    
    estimators = [M_EM, M_Spectral, M_polymom, M_true]
    estimators = [M_EM, M_Spectral, M_polygmm, M_true]
    totalerror = Counter()
    totalerrorC = Counter()

    for j in xrange(numtrials):
        gm = models.GaussianMixtures.GaussianMixtureModel.generate(k, d, means=typemean, cov=typecov, gaussian_precision=1)
        X = gm.sample(numsamp)
        print 'finished sampling'
        Mstar = gm.means.T

        for i,theta_hat in enumerate(estimators):
            Mstar = gm.means.T
            Clist = []
            for covmat in gm.sigmas:
                Clist += [sc.diag(covmat).tolist()]
            Cstar = sc.row_stack(Clist)
            
            M_,C_ = theta_hat(gm, X)

            M_ = closest_permuted_matrix(Mstar, M_)
            #Vstar = Mstar[:,0]
            #V_ = closest_permuted_vector(Vstar, M_[:,0])
            print M_
            #totalerror[theta_hat.func_name] += norm( Mstar - M_ )**2 / numtrials
            totalerror[theta_hat.func_name] += column_rerr( Mstar, M_ ) / numtrials
        
            ## if C_ is not None:
            ##     C_ = closest_permuted_matrix(Cstar, C_)
            ##     #print C_
            ##     totalerrorC[theta_hat.func_name] += norm( Cstar - C_ )**2 / numtrials
            #relerr = norm( Mstar - M_ )/norm(Mstar)
            print '%s: %.5f' % (theta_hat.func_name, norm( Mstar - M_ )**2)
        print '___'
    print 'k=%d\td=%d\tnumsamp=%d\tmean=%s\tcov=%s\tnumtrial=%d' % \
      (k,d,numsamp, typemean, typecov, numtrials)
    print args
    print totalerror

    with open('resultsgmm', 'a') as f:
        f.write('\n' + str(args) + '\n' + str(totalerror) + '\n')
        f.write('%.2f & %.2f & %.2f &\n' % (totalerror['M_EM'], totalerror['M_Spectral'], totalerror['M_polygmm']))
        f.write('****************\n')
    #print totalerrorC
    print gm.sigmas
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser( description='' )
    parser.add_argument( '--seed', type=int, default=0, help="" )
    parser.add_argument( '--N', type=float, default=1e4, help="" )
    parser.add_argument( '--trials', type=int, default=1, help="" )
    parser.add_argument( '--k', type=int, default=3, help="" )
    parser.add_argument( '--d', type=int, default=3, help="" )
    parser.add_argument( '--typemean', type=str, default='hypercube', help="hypercube,random,rotatedhypercube" )
    parser.add_argument( '--typecov', type=str, default='diagonal', help="diagonal, spherical" )
    #parser.set_defaults(func=do_command)

    #subparsers = parser.add_subparsers()
    #command_parser = subparsers.add_parser('command', help='' )
    #command_parser.set_defaults(func=do_command)

    args = parser.parse_args()
    #ARGS.func(ARGS)

    test_all_methods(args)


        
