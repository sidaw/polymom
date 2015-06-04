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

def make_distribution(vec):
    return abs(vec) / abs(vec).sum()

def do_lasserre(model, data, maxdeg=3):
    eqns = model.empirical_moment_equations(data, maxdeg)
    syms = model.param_symbols()
    M = MomentMatrix(3, syms, morder='grevlex')
    solsdp = solvers.solve_generalized_mom_coneqp(M, eqns, None)
    #solsdp = solvers.solve_basic_constraints(M, eqns, slack=1e-5)
    sol = extractors.extract_solutions_lasserre(M, solsdp['x'], Kmax=model.k, tol=1e-5)
    #sol = extractors.extract_solutions_dreesen(M, solsdp['x'], Kmax=model.k)
    #sol = extractors.extract_solutions_dreesen_proto(M, solsdp['x'], Kmax=model.k)
    return sol

def solve_mixture_model(model, data, maxdeg=3):
    syms = model.param_symbols()
    sol = do_lasserre(model, data)
    params = array([sol[sym] for sym in syms])
    print "pre", params
    params = array([make_distribution(col) for col in params.T]).T

    # TODO(arun): get w

    return model["w"], params



