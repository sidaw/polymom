#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Examples of polynomial systems
"""

import numpy as np
import sympy as sp
from numpy import zeros, array, triu_indices_from
from sympy import xring, RR, symbols, grevlex
from util import row_normalize
import BorderBasis as BB
import util

import ipdb

class Report(object):
    """
    Represents a report of the execution. 
    """
    pass

class System(object):
    """
    Represents a model class. Defines hyper parameters. 
    Methods:
        + Construct a random instance
    """

    def standard_instance(self, *args, **kwargs):
        """
        A simple standard instance of the problem.
        """
        raise NotImplementedError()

    def random_instance(self, *args, **kwargs):
        """
        Randomized version. Assumes one of kwargs is 'seed'
        """
        raise NotImplementedError()

    @staticmethod
    def run(system, args):
        """
        Run system
        """

        if args.mode == "standard":
            R, I, V = system.standard_instance()
        else:
            R, I, V = system.random_instance()

        B = BB.BorderBasisFactory(1e-5).generate(R, I)
        V_ = B.zeros()
        return V, V_

    @staticmethod
    def get_argparse(subparsers):
        """
        Get a sub-parser
        """
        raise NotImplementedError()
        # parser = subparsers.add_parser('command', help='' )
        # parser.set_defaults(func=do_command)

class LinearSystem(System):
    def __init__(self, n_variables, n_equations):
        self.n_variables, self.n_equations = n_variables, n_equations

        syms = ['x%d' % i for i in xrange(1, n_variables+1)]
        self.R, self.syms = xring(','.join(syms), RR, order=grevlex)

    def random_instance(self, *args, **kwargs):
        M = np.random.randn(self.n_variables, self.n_equations)
        I = [sum(coeff * sym for coeff, sym in zip(coeffs, self.syms))
                for coeffs in M]
        V = [np.zeros(self.n_variables)]

        return self.R, I, V

    @staticmethod
    def run(ARGS):
        """
        Initialize the linear system and run
        """
        V_ = System.run(LinearSystem(ARGS.n, ARGS.m), ARGS)

    @staticmethod
    def get_argparse(subparsers):
        """
        Get a sub-parser
        """
        parser = subparsers.add_parser('linear', help='Linear system')
        parser.add_argument("-n", default=10, help = "Number of variables")
        parser.add_argument("-m", default=10, help = "Number of equations")
        parser.set_defaults(func=LinearSystem.run)

        return parser

class Cholesky(System):
    def __init__(self, d, opt):
        self.d = d
        self.opt = opt

        l_vars = [(i,j) for i in xrange(d) for j in xrange(i+1)]
        syms = ["l_%d%d"% (i+1,j+1) for i, j in l_vars]
        self.R, self.syms = xring(",".join(syms), RR, order=grevlex)
        self.l_vars = dict(zip(l_vars, self.syms))

    def matrix_to_polys(self, M):
        d, l = self.d, self.l_vars
        def eqn(i,j):
            return sum(l.get((i,k), 1 if i == k else 0) * l.get((j,k), 0) for k in xrange(min(i,j)+1))
        I = [eqn(i,j) - M[i,j] for i in xrange(d) for j in xrange(i+1)]

        if self.opt == 1:
            I += [l.get((0,0)) - np.sqrt(M[0,0])]

        return I

    def standard_instance(self, *args, **kwargs):
        M = np.eye(self.d)
        L = M

        return self.R, self.matrix_to_polys(M), L

    def random_instance(self, *args, **kwargs):
        U = util.orthogonal(self.d)
        D = np.random.rand(self.d)
        M = U.dot(np.diag(D)).dot(U.T)
        L = np.linalg.cholesky(M)

        return self.R, self.matrix_to_polys(M), L

    @staticmethod
    def run(ARGS):
        """
        Initialize the linear system and run
        """
        V_ = System.run(Cholesky(ARGS.d, ARGS.o), ARGS)

    @staticmethod
    def get_argparse(subparsers):
        """
        Get a sub-parser
        """
        parser = subparsers.add_parser('chol', help='Linear system')
        parser.add_argument("-d", default=3, help = "Dimension of system")
        parser.add_argument("-o", choices=[0,1], default=1, help = "0 = naive, 1 = with quadratic system")
        parser.set_defaults(func=Cholesky.run)

        return parser

class Eigen(System):
    def __init__(self, d, opt):
        self.d = d
        self.opt = opt

        syms = ["l"] + ["x_%d" % (i+1) for i in xrange(d)]
        self.R, self.syms = xring(",".join(syms), RR, order=grevlex)

    def matrix_to_polys(self, M):
        d, opt = self.d, self.opt
        l, xs = self.syms[0], self.syms[1:]
        # Construct equations
        def eqn(i):
            return sum(M[i,j] * xs[j] for j in xrange(d))
        I = [eqn(i) - l * xs[i] for i in xrange(d)]

        if opt == 0:
            I += [ sum(xs[i]**2 for i in xrange(d)) - 1.0 ]
        elif opt == 1:
            I += [ xs[0] - 1.0 ]

        return I

    def standard_instance(self, *args, **kwargs):
        d = self.d
        M = np.diag(np.arange(1., d+1.))/d
        print M

        return self.R, self.matrix_to_polys(M), np.diag(M)

    def random_instance(self, *args, **kwargs):
        U = util.orthogonal(self.d)
        D = np.random.rand(self.d)
        M = U.dot(np.diag(D)).dot(U.T)

        print D, M

        return M, self.R, self.matrix_to_polys(M), D

    @staticmethod
    def run(ARGS):
        """
        Initialize the linear system and run
        """
        V_ = System.run(Eigen(ARGS.d, ARGS.o), ARGS)

    @staticmethod
    def get_argparse(subparsers):
        """
        Get a sub-parser
        """
        parser = subparsers.add_parser('eigen', help='Linear system')
        parser.add_argument("-d", default=3, help = "Dimension of system")
        parser.add_argument("-o", choices=[0,1], default=1, help = "0 = naive, 1 = with quadratic system")
        parser.set_defaults(func=Eigen.run)

        return parser

def main():
    import sys, argparse
    parser = argparse.ArgumentParser( description='Solve polynomial systems' )
    parser.add_argument('--mode', choices=["standard","random"], default="standard", help="How to generate instance")
    parser.add_argument('--seed', default=42, help="Random seed")
    parser.add_argument('--report', type=bool, default=False, help="Generate a report" )
    subparsers = parser.add_subparsers()

    import inspect
    for name, cls in inspect.getmembers(sys.modules[__name__]):
        if not inspect.isclass(cls) or not issubclass(cls, System) or cls == System:
            continue
        print name, cls
        print "Adding", name
        cls.get_argparse(subparsers)

    #subparsers = parser.add_subparsers()
    #command_parser = subparsers.add_parser('command', help='' )
    #command_parser.set_defaults(func=do_command)

    ARGS = parser.parse_args()
    np.random.seed(ARGS.seed)
    ARGS.func(ARGS)


if __name__ == "__main__":
    main()

