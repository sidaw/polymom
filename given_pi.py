#!/usr/bin/env python2.7
"""
Test when given pi
"""

import random
from sympy import symbols, poly
import numpy as np
from util import tuple_add

import polyopt

def generate_moments(hyper, params):
    """
    Measure the moments by substituting the parameters
    """

    k, d = hyper['k'], hyper['d']

    p = params # Shorthand, don't judge
    m = {} # Moments
    for x1 in xrange(1,d+1):
        m[(x1,)] = sum( p[(h,x1)] * p[(h,)] for h in xrange(1,k+1) )
        for x2 in xrange(1,d+1):
            m[(x1,x2)] = sum( p[(h,x1)] * p[(h,x2)] * p[(h,)] for h in xrange(1,k+1) )
            for x3 in xrange(1,d+1):
                m[(x1,x2,x3)] = sum( p[(h,x1)] * p[(h,x2)] * p[(h,x3)] * p[(h,)] for h in xrange(1,k+1) )
    return m

def generate_poly(hyper, params):
    """
    Create a symbolic polynomial for the moments.
    """

    k, d = hyper['k'], hyper['d']

    atoms = {}
    for h in xrange(1,k+1):
        atoms.update({ 
                (h,x1) : symbols('x_%d%d'%(h,x1))
                    for x1 in xrange(1,d+1)
                })
        #atoms[(h,d)] = 1. - sum(symbols('x_%d%d'%(h,x1)) for x1 in xrange(1,d))

    m = {}
    for x1 in xrange(1,d+1):
        m[(x1,)] = poly( sum( atoms[(h,x1)] * params[(h,)] for h in xrange(1,k+1) ) )
        for x2 in xrange(1,d+1):
            m[(x1,x2)] = poly( sum( atoms[(h,x1)] * atoms[(h,x2)] * params[(h,)] for h in xrange(1,k+1) ) )
            for x3 in xrange(1,d+1):
                m[(x1,x2,x3)] = poly( sum( atoms[(h,x1)] * atoms[(h,x2)] * atoms[(h,x3)] * params[(h,)] for h in xrange(1,k+1) ) )

    return m

def random_dist(n):
    """Random probability distribution"""
    dist = [random.random() for _ in xrange(n)]
    z = sum(dist)
    return [pr/z for pr in dist]

def generate_parameters(hyper):
    """Randomly initialize parameters"""
    k, d = hyper['k'], hyper['d']

    params = {}

    pi = random_dist(k)
    for h in xrange(1,k+1):
        params[(h,)] = pi[h-1]
        # Random choice
        o = random_dist(d)
        for x1 in xrange(1,d+1):
            params[(h,x1)] = o[x1-1]
    return params

def create_mom_poly(p,m):
    """Create the sos polynomial"""
    # polynomial.
    pol = 1.
    for key in p.keys():
        pol += (p[key] - m[key])**2
    return pol

def do_generate(args):
    K, D = 2, 2
    hyper = {
            'k' : K,
            'd' : D,
            }
    params = {
            (1,) : 0.4,
            (2,) : 0.6,
            (1,1) : 1,
            (1,2) : 0,
            (2,1) : 0,
            (2,2) : 1,
            }

    params = generate_parameters(hyper)
    params_pol = generate_poly(hyper, params)
    mom = generate_moments(hyper, params)
    pol = create_mom_poly(params_pol,mom)
    sol = polyopt.optimize_polynomial(pol)

    m = pol.monoms()

    print zip(m[:(K+K*D)], sol['x'][:(K+K*D)])
    return sol

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser( description='Does the polynomial optimization procedure work when we are given the values of $\pi$ (hence breaking symmetry)?' )
    parser.set_defaults(func=do_generate)

    ARGS = parser.parse_args()
    ARGS.func(ARGS)

