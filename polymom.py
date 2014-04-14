#!/usr/bin/env python2.7
"""
Learning latent variable models using polynomial optimization
"""

import random
from sympy import symbols, poly
import numpy as np
from util import tuple_add

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
    atoms = { 
                (h,) : symbols('h_%d'%h)
                for h in xrange(1, k)
            }
    atoms[(k,)] = 1. - sum( symbols('h_%d'%h) for h in xrange(1, k) )

    for h in xrange(1,k+1):
        atoms.update({ 
                (h,x1) : symbols('x_%d%d'%(h,x1))
                    for x1 in xrange(1,d)
                })
        atoms[(h,d)] = 1. - sum(symbols('x_%d%d'%(h,x1)) for x1 in xrange(1,d))

    m = {}
    for x1 in xrange(1,d+1):
        m[(x1,)] = poly( sum( atoms[(h,x1)] * atoms[(h,)] for h in xrange(1,k+1) ) )
        for x2 in xrange(1,d+1):
            m[(x1,x2)] = poly( sum( atoms[(h,x1)] * atoms[(h,x2)] * atoms[(h,)] for h in xrange(1,k+1) ) )
            for x3 in xrange(1,d+1):
                m[(x1,x2,x3)] = poly( sum( atoms[(h,x1)] * atoms[(h,x2)] * atoms[(h,x3)] * atoms[(h,)] for h in xrange(1,k+1) ) )

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
    basis = set([])
    pol = 1.
    for key in p.keys():
        basis.update(p[key].monoms())
        pol += (p[key] - m[key])**2

    # get basis.
    basis = sorted(basis)
    return basis, pol

def generate_cvx(A):
    """Generate the CVX program from the matrix of coefficients A"""
    template = """
# Auto-generated script 
# TODO: populate program
cvx_begin

cvx_end

"""
    print template

def do_generate(args):
    hyper = { 'k': args.k, 'd' : args.d, 'v' : args.v }

    # TODO: Support arbitrary number of moments.
    if hyper['v'] != 3: 
        raise NotImplementedError() 

    params = generate_parameters(hyper)
    p = generate_poly(hyper, params)
    m = generate_moments(hyper, params)
    basis, pol = create_mom_poly(p,m)
    A = find_coeffs(basis, pol)

    print A

    generate_cvx(A)

    return basis, pol, A

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser( description='Generate moment constraints for a three-view mixture model' )
    parser.add_argument( '-k', type=int, default=2, help="Latent dimension" )
    parser.add_argument( '-d', type=int, default=2, help="Observed dimension" )
    parser.add_argument( '-v', type=int, default=3, help="Number of moments to look at" )
    parser.set_defaults(func=do_generate)

    ARGS = parser.parse_args()
    ARGS.func(ARGS)
