#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Algorithms that use the tensor power method as a routine.
"""

import sympy as sp

import ipdb

from algos.tensor_power_method import candecomp
from util import symmetrize, symmetric_skew, get_whitener, dict_diff

import scipy as sc
from scipy import diag, sqrt, zeros, einsum, array
from scipy.linalg import norm, svdvals, eig, eigvals, pinv, cholesky, inv
from util import fix_parameters, normalize

from models.MixtureModel import MixtureModel

from nose import with_setup

def make_distribution(vec):
    return abs(vec) / abs(vec).sum()

def solve_mixture_model(model, data):
    """
    Whiten and unwhiten appropriately
    """

    d = model["d"]

    # Get moments
    moments = model.empirical_moments(data, model.observed_monomials(3))
    M2 = zeros((d, d))
    M3 = zeros((d, d, d))

    for i in xrange(d):
        for j in xrange(d):
            xij = sp.sympify('x%d * x%d' %(i+1, j+1))
            M2[i,j] = moments[xij]

            for k in xrange(d):
                xijk = sp.sympify('x%d * x%d * x%d' % (i+1, j+1, k+1))
                M3[i,j,k] = moments[xijk]

    k = model["k"]
    # Symmetrize
    M2, M3 = symmetrize(M2), symmetrize(M3)

    assert symmetric_skew(M2) < 1e-2
    assert symmetric_skew(M3) < 1e-2

    # Whiten
    W, Wt = get_whitener(M2, k)
    M3_ = einsum('ijk,ia,jb,kc->abc', M3, W, W, W)

    pi_, M_, _, _ = candecomp(M3_, k)

    # Unwhiten M
    M_ = Wt.dot(M_.dot(diag(pi_)))
    pi_ = 1./pi_**2
    # "Project" onto simplex
    pi_ = make_distribution(abs(pi_))
    M_ = array([make_distribution(col) for col in M_.T]).T

    return pi_, M_

def set_random_seed():
    """
    Setup hook for test functions
    """
    sc.random.seed(42)

@with_setup(set_random_seed)
def test_mom():
    """Test the whether spectral can recover from simple mixture of 3 identical"""
    model = MixtureModel.generate(2, 3, dirichlet_scale = 0.9)
    _, _, M, _ = model["k"], model["d"], model["M"], model["w"]
    N = 1e5

    xs = model.sample(N)

    moments = model.exact_moments(model.observed_monomials(3))
    moments_ =  model.empirical_moments(xs, model.observed_monomials(3))

    print "moment diff", dict_diff(moments, moments_)

    w_true, params = solve_mixture_model(model, moments)
    w_data, params_ = solve_mixture_model(model, moments_)

    w_true, params = fix_parameters(M, params, w_true)
    w_data, params_ = fix_parameters(M, params_, w_data)

    print "true", M
    print "with exact moments", params
    print "error with exact moments", norm(M - params)/norm(M)
    print "with empirical moments", params_
    print "error with empirical moments", norm(M - params_)/norm(M)

    assert norm(M - params)/norm(M) < 1e-10
    assert norm(M - params_)/norm(M) < 1e-1

