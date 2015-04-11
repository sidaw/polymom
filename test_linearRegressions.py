#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""

"""

import numpy as np
from models import LinearRegressionsMixture
from util import prod, avg, partitions

N = 1e6

def get_model():
    model = LinearRegressionsMixture.generate("temp.dat", 2, 2)
    print "True parameters:",
    print model.betas

    ys, xs = model.sample(N)
    return ys, xs

def coefficients(xs, alpha, b):
    r"""
    Compute the coefficients of w^{‥} in
    $x^{\alpha + \bar i} \sum_{i=1}^d w\oftt{k}{\bar i}$
    """

    d = len(alpha)

    betas = list(partitions(d, b))
    coeffs = np.zeros(len(betas))
    for i, (y, x) in enumerate(zip(ys, xs)):
        for j, beta in enumerate(betas):
            coeffs[j] += (prod(xi**(a+b) for xi, a, b in zip(x, alpha, beta)) - coeffs[j])/(i+1)

    return coeffs  

def compute_moments(ys, xs, alpha, b):
    r"""
    Compute $\E[x^\alpha y^b] = \E[ \sum_{k=1}^K \pi_k \sum_{i=1}^d w\oftt{k}{\bar i} x^{\alpha + \bar i}]$.
    """
    
    return avg(prod(xi**a for xi, a in zip(x, alpha)) * y**b for y, x in zip(ys, xs))


