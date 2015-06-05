#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""

"""
import ipdb

import numpy as np
from numpy import array
from numpy.linalg import norm
from util import fix_parameters, column_rerr
import csv
import sys
from models.MixtureModel import MixtureModel
from algos import em, spectral

from mompy.core import MomentMatrix
import mompy.solvers as solvers
import mompy.extractors as extractors

from itertools import combinations, combinations_with_replacement
import sympy as sp

def make_distribution(vec):
    return abs(vec) / abs(vec).sum()

def get_sumto1constraints(syms, maxdeg = 4):
    if maxdeg < 0: return []
    P = len(syms)
    sum1 = -1
    for sym in syms:
        sum1 = sum1 + sym

    sum1eqs = []
    for j in range(1,maxdeg):
        slices = combinations_with_replacement(range(P), j)
        for s in slices:
            currenteq = sum1
            for i in s:
                currenteq = currenteq * syms[i]
            sum1eqs.append(sp.expand(currenteq))

    return sum1eqs

            
def do_lasserre(model, data, maxdeg=3, maxcontrs=-1):
    eqns = model.empirical_moment_equations(data, 3)
    syms = model.param_symbols()
    #ipdb.set_trace()    
    M = MomentMatrix(maxdeg, syms, morder='grevlex')
    
    solsdp = solvers.solve_generalized_mom_coneqp(M, eqns, None)
    #solsdp = solvers.solve_basic_constraints(M, eqns, slack=1e-5)
    sol = extractors.extract_solutions_lasserre(M, solsdp['x'], Kmax=model.k, tol=1e-5)
    #sol = extractors.extract_solutions_dreesen(M, solsdp['x'], Kmax=model.k)
    #sol = extractors.extract_solutions_dreesen_proto(M, solsdp['x'], Kmax=model.k)
    return sol

def solve_mixture_model(model, data, maxdeg=3, maxcontrs=-1):
    syms = model.param_symbols()
    sol = do_lasserre(model, data, maxdeg, maxcontrs)
    params = array([sol[sym] for sym in syms])
    print "pre", params
    params = array([make_distribution(col) for col in params.T]).T

    # TODO(arun): get w

    return model["w"], params



