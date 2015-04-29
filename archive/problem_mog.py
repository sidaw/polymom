#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
The mixture of Gaussians problem
"""

import ipdb
import numpy as np
from sympy import symbols, poly
from polyopt import *

def random_dist(n):
    """Random probability distribution"""
    dist = [np.random.randint(1,10) for _ in xrange(n)]
    z = 1. #sum(dist)
    return [pr/z for pr in dist]

def sym(h, x):
    return 'x_%d%d'%(h,x)

def generate_parameters(hyper):
    """Randomly initialize parameters"""
    k, d = hyper['k'], hyper['d']

    params = {}

    for h in xrange(1,k+1):
        # Random choice
        o = random_dist(d)
        for x1 in xrange(1,d+1):
            params[sym(h,x1)] = o[x1-1]
    return params

def generate_poly(hyper, syms):
    """
    Create a symbolic polynomial for the moments.
    """
    k, d = hyper['k'], hyper['d']

    atoms = {}
    for h in xrange(1,k+1):
        atoms.update({ 
                (h,x1) : symbols(sym(h,x1))
                    for x1 in xrange(1,d+1)
                })

    m = {}
    for x1 in xrange(1,d+1):
        m[(x1,)] = poly( sum( atoms[(h,x1)] for h in xrange(1,k+1) ), *syms, domain='RR')
        for x2 in xrange(x1,d+1):
            m[(x1,x2)] = poly( sum( atoms[(h,x1)] * atoms[(h,x2)] for h in xrange(1,k+1) ), *syms, domain='RR')
#            for x3 in xrange(x2,d+1):
#                 m[(x1,x2,x3)] = poly( sum( atoms[(h,x1)] * atoms[(h,x2)] * atoms[(h,x3)] for h in xrange(1,k+1) ), *syms, domain='RR')

    return m

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

def get_problem_instance(k=2, d=2, seed=0):
    hyper = {'k' : k, 'd' : d}

    params = generate_parameters(hyper)
    print "params", params
    syms = [symbols(p) for p in params.keys()]

    I_ = generate_poly(hyper, syms)
    print {m : p.eval(params) for m, p in I_.items()}

    # Evaluate moments
    I = [p - p.eval(params) for p in I_.values()]
    I = I[:-3]
    print I

    return I, syms, params

def do_command(args):
    I, syms, params = get_problem_instance(k=ARGS.k, d=ARGS.d, seed=ARGS.seed)
    sol = solve_companion_matrix(I, *syms)
    print "Solution (sym, vals). Values are aligned"
    for sym, vals in sol.items():
        print sym, vals, params[str(sym)]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser( description='Solve the 2x2 problem' )
    parser.add_argument( '--seed', type=int, default=0, help="Random seed used to generate hte problem" )
    parser.add_argument( '--k', type=int, default=2, help="Random seed used to generate hte problem" )
    parser.add_argument( '--d', type=int, default=2, help="Random seed used to generate hte problem" )
    parser.set_defaults(func=do_command)

    #subparsers = parser.add_subparsers()
    #command_parser = subparsers.add_parser('command', help='' )
    #command_parser.set_defaults(func=do_command)

    ARGS = parser.parse_args()
    np.random.seed(ARGS.seed)
    ARGS.func(ARGS)
