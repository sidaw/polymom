#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Given a model and list of algorithms, run them and print output.
"""
import ipdb

import numpy as np
from numpy import array
from numpy.linalg import norm
import util
from util import fix_parameters, column_rerr
import csv
import sys
from models.MixtureModel import MixtureModel
from algos import em, spectral, lasserre

from mompy.core import MomentMatrix
import mompy.solvers as solvers
import mompy.extractors as extractors

from collections import Counter

def make_distribution(vec):
    return abs(vec) / abs(vec).sum()

def do_dreesen(model, data, maxdeg=3):
    eqns = model.empirical_moment_equations(data, maxdeg)
    syms = model.param_symbols()
    M = MomentMatrix(2, syms, morder='grevlex')
    solsdp = solvers.solve_generalized_mom_coneqp(M, eqns, None)
    sol = extractors.extract_solutions_lasserre(M, solsdp['x'], Kmax=model.k)
    params = array([sol[sym][0] for sym in syms])
    params = array([make_distribution(col) for col in params.T]).T

    return model["w"], params

def do_lasserre(model, data, maxdeg=3, maxcontrs=-1):
    if isinstance(model, MixtureModel):
        return lasserre.solve_mixture_model(model, data, maxdeg, maxcontrs=-1)

def do_lasserreconstr(model, data, maxdeg=3, maxcontrs=6):
    if isinstance(model, MixtureModel):
        return lasserre.solve_mixture_model(model, data, maxdeg, maxcontrs = maxcontrs)

def do_tpm(model, data):
    if isinstance(model, MixtureModel):
        return spectral.solve_mixture_model(model, data)

def do_em(model, data, iters = 1):
    if isinstance(model, MixtureModel):
        em_fn = em.solve_mixture_model
    else:
        raise NotImplementedError

    w_em, params_em, lhood_em = None, None, -np.inf
    for _ in xrange(iters):
        w_em_, params_em_ = em_fn(model, data)
        lhood_em_ = model.using(M=params_em_, w=w_em_).llikelihood(data)
        if lhood_em_ > lhood_em:
            w_em, params_em, lhood_em = w_em_, params_em_, lhood_em_
    return w_em, params_em

def make_table(model, data, methods):
    tbl = []
    tbl.append(["Method","Parameter error", "Likelihood"])
    tbl.append(["True", 0., model.llikelihood(data)])
    print "true", model["M"]
    for name, method in methods:
        w, params = method(model, data)
        w, params = fix_parameters(model["M"], params, w)

        print name, params
        lhood = model.using(M=params, w=w).llikelihood(data)
        tbl.append([name, column_rerr(model["M"], params), lhood])
    return tbl

def print_table(arr):
    writer = csv.writer(sys.stdout)
    for row in arr:
        writer.writerow(row)

def do_command(args):
    np.random.seed(args.seed)
    #methods = [("EM", do_em), ("TPM", do_tpm), ("Lasserre", do_lasserre), ("Dreesen", do_dreesen)]
    #methods = [("EM", do_em), ("TPM", do_tpm),("Lasserre", do_lasserre)] #, ("Dreesen", do_dreesen)]
    methods = [("EM", do_em), ("TPM", do_tpm),("Lasserre", do_lasserre), ("Lasserreconstr", do_lasserreconstr)] #, ("Dreesen", do_dreesen)]
    avg_paramserror = Counter(); avg_nll = Counter();
    for i in xrange(args.trials):
        model = MixtureModel.generate(k = args.k, d = args.d)
        data = model.sample(int(args.N))
        tbl = make_table(model, data, methods)
        for mname,_ in methods:
            for row in tbl:
                if row[0] == mname:
                    avg_paramserror[mname] += row[1]/args.trials
                    avg_nll[mname] += row[2]/args.trials
            
        print_table(tbl)
        print avg_paramserror
        print avg_nll
        print args

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser( description='' )
    parser.add_argument( '--seed', type=int, default=0, help="" )
    parser.add_argument( '--N', type=float, default=1e4, help="" )
    parser.add_argument( '--trials', type=int, default=1, help="" )
    parser.add_argument( '--k', type=int, default=3, help="" )
    parser.add_argument( '--d', type=int, default=3, help="" )
    
    parser.set_defaults(func=do_command)

    #subparsers = parser.add_subparsers()
    #command_parser = subparsers.add_parser('command', help='' )
    #command_parser.set_defaults(func=do_command)

    ARGS = parser.parse_args()
    ARGS.func(ARGS)
