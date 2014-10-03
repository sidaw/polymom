#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
The 2x2 problem
"""

import numpy as np
from polyopt import *

def get_problem_instance(seed=0):
    x1, x2, y1, y2 = sp.symbols('x1,x2,y1,y2')
    syms = [x1, x2, y1, y2]
    m11 = sp.poly(x1 + x2, *syms, domain='RR')
    m12 = sp.poly(y1 + y2, *syms, domain='RR')
    m211 = sp.poly(x1**2 + x2**2, *syms, domain='RR')
    m212 = sp.poly(x1 * y1 + x2 * y2, *syms, domain='RR')
    m222 = sp.poly(y1**2 + y2**2, *syms, domain='RR')
    m3111 = sp.poly(x1**3 + x2**3, *syms, domain='RR')
    m3112 = sp.poly(x1**2 * y1 + x2**2 * y2, *syms, domain='RR')
    m3122 = sp.poly(x1 * y1**2 + x2 * y2**2, *syms, domain='RR')
    m3222 = sp.poly(y1**3 + y2**3, *syms, domain='RR')
    ms = [ m11, m12, m211, m212, m222, m3111, m3112, m3122, m3222, ]

    if seed == 0:
        x1v, x2v = 1, -1
        y1v, y2v = -1, 1
    else:
        x1v, x2v = np.random.randint(-10,10), np.random.randint(-10,10)
        y1v, y2v = np.random.randint(-10,10), np.random.randint(-10,10)
    print "x1: ", x1v
    print "x2: ", x2v
    print "y1: ", y1v
    print "y2: ", y2v

    I = [ m - m(x1v, x2v, y1v, y2v) for m in ms]
    print "Ideal: ", I

    return I, syms

def do_command(args):
    I, syms = get_problem_instance(ARGS.seed)
    sol = solve_companion_matrix(I, *syms)
    print "Solution (sym, vals). Values are aligned"
    for sym, vals in sol.items():
        print sym, vals

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser( description='Solve the 2x2 problem' )
    parser.add_argument( '--seed', type=int, default=0, help="Random seed used to generate hte problem" )
    parser.set_defaults(func=do_command)

    #subparsers = parser.add_subparsers()
    #command_parser = subparsers.add_parser('command', help='' )
    #command_parser.set_defaults(func=do_command)

    ARGS = parser.parse_args()
    np.random.seed(ARGS.seed)
    ARGS.func(ARGS)
