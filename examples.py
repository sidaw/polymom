#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Examples of graphical models and associated polynomials
"""

import numpy as np
import sympy as sp
from numpy import zeros, array, triu_indices_from
from sympy import ring, RR, symbols
from util import row_normalize

import ipdb

class Model(object):
    """
    Represents a model class. Defines hyper parameters. 
    Methods:
        + Construct a random instance
    """

    def random_instance(self, *args):
        raise NotImplementedError()

class ModelInstance(object):
    """
    Represents a model instance. Has specific parameters.
    Methods:
        + Has parameters
        + (Draw samples)
        + Construct moment polynomials (with measurement noise)
    """

    def __init__(self, model):
        self.__model = model

#    @attribute
    def get_parameters(self):
        raise NotImplementedError()

#    @attribute
    def get_measurement_polynomials(self, noise = 0, seed = 0):
        raise NotImplementedError()

    def draw_samples(self):
        raise NotImplementedError()


class BernoulliMixture(Model):
    """
    Represents a mixture of k bernoullis.
    """

    def __init__(self):
        pass

    class Instance(ModelInstance):
        def __init__(self, model, params):
            ModelInstance.__init__(self, model)
            self.d, self.k = params.shape
            self.params = params

        def __repr__(self):
            return "[BernoulliMixture (%d, %d)]"%(self.k, self.d)


        def x(self, d, k):
            return "x%d_%d"%(d,k)

#        @attribute
        def get_parameters(self):
            return [(self.x(i,j), self.params[i,j])
                    for i in xrange(self.d) for j in xrange(self.k)]

#        @attribute
        def get_measurement_polynomials(self, noise = 0, seed = 0):
            np.random.seed(seed)

            k, d = self.k, self.d
            params = self.get_parameters()
            R = ring([x for x, _ in params], RR)[0]
            names = {str(x) : R(x) for x in R.symbols}
            xs = array([[names[self.x(i,j)] for j in xrange(k)] for i in xrange(d)])
            params = [(names[x], v) for x, v in params]

            # Second order moments (TODO: 3rd order moments)
            P = zeros((d,d), dtype=np.object)
            p = zeros((d,), dtype=np.object)
            for i in xrange(d):
                p[i] = sum(xs[i,k_] for k_ in xrange(k))# / k
                for j in xrange(i, d):
                    P[i,j] = sum(xs[i,k_] * xs[j,k_] for k_ in xrange(k))# / k

            # Project and profit
            m = zeros((d,))
            M = zeros((d,d))
            for i in xrange(d):
                m[i] = p[i].evaluate(params)
                for j in xrange(i, d):
                    M[i,j] = P[i,j].evaluate(params)
            M = M + noise * np.random.randn(d,d)
            m = m + noise * np.random.randn(d)
            # TODO: Something is wrong here 
            #m = M.sum(1)

            # Finally return values.
            return R, [f - f_ 
                    for f, f_ in zip(p.flatten(), m.flatten())] + [f - f_ 
                            for f, f_ in zip(P[triu_indices_from(P)], M[triu_indices_from(M)])]


    def random_instance(self, k, d, *args):
        """
        Hyper parameters: d, k, beta
        """
        # Row normalize
        params = abs(np.random.rand(d,k))
        params = row_normalize(params.T, 1).T

        return BernoulliMixture.Instance(self, params)

