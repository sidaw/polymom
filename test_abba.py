#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Test the border basis algorithm.
"""

import numpy as np
import sympy as sp

from sympy import ring, xring, RR, lex, grlex, grevlex, pprint, Eq, Or
from numpy import sin, cos, pi, array
from numpy.random import rand, randn
from numpy.linalg import norm

import BorderBasis as BB
from util import prod, closest_permuted_vector

def poly_for_zeros(R, x, Z):
    return prod(x - z for z in Z)

def generate_univariate_problem(n_common_zeros = 1, max_degree = 3, n_equations = 5):
    R, x = ring('x', RR, order=grevlex)
    # set of common zeros.
    V = 1 - 2*rand(n_common_zeros)
    I = [poly_for_zeros(R, x, np.hstack((V, 1 - 2*rand(max_degree - n_common_zeros)))) for _ in xrange(n_equations)]
    return R, I, V

def add_noise(I, sigma = 0.):
    return [i + sigma * randn() for i in I]

def get_stats(n=3, d=20, e=20, sigma=0., tries=100):
    status, epses, err = [], [], []
    for _ in xrange(tries):
        R, I, V = generate_univariate_problem(n, d, e)
        I = add_noise(I, sigma)
        for eps in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
            try :
                V_ = BB.BorderBasisFactory(sigma + eps).generate(R, I).zeros()
                V_ = array(V_).flatten()
                epses.append(eps)
                break
            except AssertionError:
                continue
        else:
            status.append(1)
            epses.append(1)
            err.append(1)
            continue
        if len(V_) < len (V):
            status.append(2)
            # Pad V_.
            V_ = np.hstack((V_, np.zeros(len(V) - len(V_))))
        elif len(V_) > len (V):
            status.append(3)
            # Pad V_.
            V = np.hstack((V, np.zeros(len(V_) - len(V))))
        else:
            status.append(4)

        V_ = closest_permuted_vector(V, V_)
        err.append(norm(V - V_))
    return status, epses, err
    
def test_univarate(n=3, d=20, e=20):
    #for sigma in [0, 1e-4, 1e-3, 1e-2]:
    for _ in xrange(100):
        sigma = 0
        R, I, V = generate_univariate_problem(n, d, e)
        print "V", V
        I = add_noise(I, sigma)
        for eps in [1e-7, 1e-6, 1e-5, 1e-4]:
            try :
                V_ = BB.BorderBasisFactory(eps + sigma).generate(R, I).zeros()
                # Find the maximal matching between V and V_
                V_ = array(V_).flatten()
                print "V_", V_
                print 'e', eps
                break
            except AssertionError:
                continue
        else:
            print "could not find 0"
            return R, I, V
        if len(V) == len(V_):
            V_ = closest_permuted_vector(V, V_)
            print "diff", (V - V_)
            if max(abs(V - V_)) > max(1e-3, sigma):
                print "error too large!"
                return R, I, V
        else:
            print "incorrect lengths!"
            return R, I, V

def generate_multivariate_problem(n_common_zeros = 1, n_variables=10, n_equations=10):
    syms = ['x%d'%i for i in xrange(1,n_variables+1)]
    R, syms = xring(','.join(syms), RR, order=grevlex)

    I = [sum(coeff * sym for coeff, sym in zip(coeffs, syms))
            for coeffs in np.random.randn(n_variables, n_equations)]
    return R, I, [np.zeros(n_variables)]

