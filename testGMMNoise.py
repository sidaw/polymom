# coding: utf-8

# # Mixture of Gaussians

import util
from util import *
import numpy as np
import models
import mompy as mp
import cvxopt
import sys
from operator import mul
import sympy as sp
from collections import Counter

from cvxopt import solvers
solvers.options['show_progress'] = False

def M_polymom(gm, X, degmm=3, degobs=4):
    tol= 1e-2
    k = gm.k
    
    monos = gm.polymom_monos(degmm)
    constraints = gm.polymom_all_constraints(degobs)
    xis = gm.sym_means
    covs = gm.sym_covs
    sym_all = xis + covs
    MM = mp.MomentMatrix(degmm, sym_all, morder='grevlex', monos=monos)
    constraints_noisy = gm.polymom_all_constraints_samples(degobs, X)

    cin = mp.solvers.get_cvxopt_inputs(MM, constraints_noisy)

    for i in xrange(1):
        randdir =cvxopt.matrix( np.random.rand(*cin['c'].size) )
        solsdp_noisy = solvers.sdp(cin['c'], Gs=cin['G'], hs=cin['h'], A=cin['A'], b=cin['b'])
        soln = np.array(solsdp_noisy['ss'][0])

        Us,Sigma,Vs=np.linalg.svd(soln)

        if Sigma[k] <= tol:
            break
        else:
            pass
            #print '%.4f' % Sigma[k]
        
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

def M_polymomconvexiter(gm, X, degmm=3, degobs=4):
    tol= 1e-2
    k = gm.k
    
    monos = gm.polymom_monos(degmm)
    constraints = gm.polymom_all_constraints(degobs)
    xis = gm.sym_means
    covs = gm.sym_covs
    sym_all = xis + covs
    MM = mp.MomentMatrix(degmm, sym_all, morder='grevlex', monos=monos)
    constraints_noisy = gm.polymom_all_constraints_samples(degobs, X)

    cin = mp.solvers.get_cvxopt_inputs(MM, constraints_noisy)

    for i in xrange(1):
        randdir =cvxopt.matrix( np.random.rand(*cin['c'].size) )
        solsdp_noisy = solvers.sdp(cin['c'], Gs=cin['G'], hs=cin['h'], A=cin['A'], b=cin['b'])
        soln = np.array(solsdp_noisy['ss'][0])

        Us,Sigma,Vs=np.linalg.svd(soln)

        if Sigma[k] <= tol:
            break
        else:
            pass
            #print '%.4f' % Sigma[k]
        
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
    sklgmm = mixture.GMM(n_components=k, covariance_type='diag', n_init=1, n_iter = 10, thresh = 1e-2)
    sklgmm.fit(X)
    return sklgmm.means_, sklgmm.covars_

def M_true(gm, X):
    M_ = sc.random.permutation(gm.means.T)
    Clist = []
    for covmat in gm.sigmas:
        Clist += [sc.diag(covmat).tolist()]
    C_ = sc.random.permutation(sc.row_stack(Clist))
    return M_, C_

def test_all_methods():
    k = 3
    d = 3
    numsamp = 50000
    typemean = 'hypercube'
    typecov = 'diagonal'
    numtrials = 10
    #sc.random.seed(101)
    
    estimators = [M_EM, M_Spectral, M_polymom, M_true]
    estimators = [M_EM,  M_Spectral, M_polymom, M_true]
    totalerror = Counter()
    totalerrorC = Counter()

    for j in xrange(numtrials):
        gm = models.GaussianMixtureModel.generate('', k, d, means=typemean, cov=typecov, gaussian_precision=1)
        X = gm.sample(numsamp)
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
            totalerror[theta_hat.func_name] += norm( Mstar - M_ )**2 / numtrials
        
            if C_ is not None:
                C_ = closest_permuted_matrix(Cstar, C_)
                #print C_
                totalerrorC[theta_hat.func_name] += norm( Cstar - C_ )**2 / numtrials
            #relerr = norm( Mstar - M_ )/norm(Mstar)
            print '%s: %.5f' % (theta_hat.func_name, norm( Mstar - M_ )**2)
        print '___'
    print 'k=%d\td=%d\tnumsamp=%d\tmean=%s\tcov=%s\tnumtrial=%d' % \
      (k,d,numsamp, typemean, typecov, numtrials)
    print totalerror
    #print totalerrorC
    print gm.sigmas

test_all_methods()


        
