#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Representing eigen-factorization as a polynomial
"""

import scipy as sc
import scipy.linalg
import numpy as np
import sympy as sp
from numpy import zeros, array, triu_indices_from
from sympy import xring, RR, symbols
from util import row_normalize
import BorderBasis as BB

import ipdb

def eigen_factorize(A):
    """
    Solves LU of A using polymom
    Assumes that diagonal elements of L are 1
    """
    d, d_ = A.shape
    assert d == d_ # Only triangular for now

    syms = ["l"] + ["x_%d" % i for i in xrange(d)]
    R, syms = xring(",".join(syms), RR)
    l, xs = syms[0], syms[1:]

    # Construct equations
    def eqn(i):
        return sum(A[i,j] * xs[j] for j in xrange(d))
    I = [eqn(i) - l * xs[i] for i in xrange(d)]
    I += [ xs[0] - 1.0 ]
    #I += [ sum(xs[i]**2 for i in xrange(d)) - 1.0 ]

    return R, I

    B = BB.BorderBasisFactory(1e-5).generate(R,I)
    V = B.zeros()

    return R, I

