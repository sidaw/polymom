#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Given a model and list of algorithms, run them and print output.
"""

import numpy as np
from numpy.linalg import norm
from util import fix_parameters, column_rerr
import csv
import sys
from models.MixtureModel import MixtureModel
from algos import em, spectral

def dreesen(model, data):
    return model.params

def print_table(arr):
    writer = csv.writer(sys.stdout)
    for row in arr:
        writer.writerow(row)

def do_command(args):
    np.random.seed(args.seed)

    model = MixtureModel.generate(k = 2, d = 2)
    data = model.sample(int(args.N))

    tbl = []
    tbl.append(["Method","Parameter error", "Likelihood"])
    tbl.append(["True", 0., model.llikelihood(data)])

    w_em, params_em, lhood_em = None, None, -np.inf
    for _ in xrange(10):
        w_em_, params_em_ = em.solve_mixture_model(model, data)
        w_em_, params_em_ = fix_parameters(model["M"], params_em_, w_em_)
        lhood_em_ = model.using(M=params_em_, w=w_em_).llikelihood(data)
        if lhood_em_ > lhood_em:
            w_em, params_em, lhood_em = w_em_, params_em_, lhood_em_

    tbl.append(["EM", column_rerr(model["M"], params_em), lhood_em])

    w_tpm, params_tpm = spectral.solve_mixture_model(model, data)
    w_tpm, params_tpm = fix_parameters(model["M"], params_tpm, w_tpm)
    tbl.append(["TPM", column_rerr(model["M"], params_tpm), model.using(M=params_tpm, w=w_tpm).llikelihood(data)])

    # TODO(sidaw): Introduce dressen

    print_table(tbl)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser( description='' )
    parser.add_argument( '--seed', type=int, default=0, help="" )
    parser.add_argument( '--N', type=float, default=1e4, help="" )
    parser.set_defaults(func=do_command)

    #subparsers = parser.add_subparsers()
    #command_parser = subparsers.add_parser('command', help='' )
    #command_parser.set_defaults(func=do_command)

    ARGS = parser.parse_args()
    ARGS.func(ARGS)
