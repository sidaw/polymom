#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
E-M algorithm to detect and separate multiview
"""
import ipdb
from nose import with_setup

from algos.EMAlgorithm import EMAlgorithm
import scipy as sc
import scipy.misc, scipy.spatial, scipy.linalg

from scipy import array, eye, ones, log, exp
from scipy.linalg import norm
cdist = scipy.spatial.distance.cdist
multivariate_normal = scipy.random.multivariate_normal
multinomial = scipy.random.multinomial
dirichlet = scipy.random.dirichlet
logsumexp = scipy.logaddexp.reduce

from util import fix_parameters, normalize_columns
from models.MixtureModel import MixtureModel

class MixtureModelEM(EMAlgorithm):
    """A mixture of multinomials"""
    def __init__(self, k, d):
        self.k, self.d = k, d
        EMAlgorithm.__init__(self)

    def compute_expectation(self, xs, params):
        """Compute the most likely values of the latent variables; returns lhood"""
        xs1, xs2, xs3 = xs[:,:,0], xs[:,:,1], xs[:,:,2]
        N = len(xs)
        M, w = params

        # Get indices where on
        x1, x2, x3 = xs1.argmax(1), xs2.argmax(1), xs3.argmax(1)


        total_lhood = 0.
        # p(x, z)
        #ipdb.set_trace()
        zs = log(w) + log(M[x1]) + log(M[x2]) + log(M[x3])
        total_lhood += logsumexp(zs, 1).sum() / N

        # Normalise the probilities (soft EM)
        zs = exp(zs.T - logsumexp(zs, 1)).T

        return total_lhood, zs

    def compute_maximisation(self, xs, zs, params):
        """Compute the most likely values of the parameters"""

        xs1, xs2, xs3 = xs[:,:,0], xs[:,:,1], xs[:,:,2]
        M, w = params

        # Cluster weights
        w = zs.sum(0)

        # Get new means
        #ipdb.set_trace()
        M = ((zs.T.dot(xs1).T / w) + (zs.T.dot(xs2).T / w) + (zs.T.dot(xs3).T / w))/3

        w /= w.sum()

        return M, w

    def run(self, xs, params = None, *args, **kwargs):
        """
        Run with default args
        """
        if params == None:
            # Randomly initialize with means near uniform.
            M = dirichlet(ones(self.d) * 1., self.k).T
            w = ones(self.k)/self.k
            params = M, w
        return EMAlgorithm.run(self, xs, params, *args, **kwargs)

def set_random_seed():
    """
    Setup hook for test functions
    """
    sc.random.seed(42)

def solve_mixture_model(model, xs):
    """
    Use EM to solve the mixture model
    """
    _, _, (params, w) = MixtureModelEM(model["k"], model["d"]).run(xs)
    return w, params

@with_setup(set_random_seed)
def test_mixture_em():
    model = MixtureModel(k = 2, d = 3, M = array([[0.7, 0.3],[0.2, 0.3],[0.1, 0.4]]), w = array([0.6,0.4]))
    k, d, M, w = model["k"], model["d"], model["M"], model["w"]
    N = 1e5

    xs = model.sample(N)

    lhood, _, (params, w_true) = MixtureModelEM(k, d).run(xs, params=(M, w))
    w_true, params = fix_parameters(M, params, w_true)

    lhood_, _, (params_, w_guess) = MixtureModelEM(k, d).run(xs)
    w_data, params_ = fix_parameters(M, params_, w_guess)

    print "true", M
    print "true lhood", model.llikelihood(xs)
    print "with em*", params
    print "em lhood*", lhood
    print "em lhood*", model.using(M = params, w = w_true).llikelihood(xs)
    print "error with em*", norm(M - params)/norm(M)
    print "with em", params_
    print "em lhood", lhood_
    print "em lhood", model.using(M = params_, w = w_guess).llikelihood(xs)
    print "error with em", norm(M - params_)/norm(M)

    assert norm(M - params)/norm(M) < 1e-1
    assert norm(M - params_)/norm(M) < 1e-1

