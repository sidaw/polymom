#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Examples of graphical models and associated polynomials
"""

class Model(object):
    """
    Represents a model class. Defines hyper parameters. 
    Methods:
        + Construct a random instance
    """
    pass

class ModelInstance(object):
    """
    Represents a model instance. Has specific parameters.
    Methods:
        + Has parameters
        + (Draw samples)
        + Construct moment polynomials (with measurement noise)
    """
    pass


class BernoulliMixture(Model):
    """
    Hyper parameters: d, k, beta
    """
    pass

