#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Global information
"""

class AlgorithmInformation(object):

    def __init__(self, L, I):
        self.L = L
        self.I = I
        self.Ws = []
        self.stage = 0
        self.iteration = 0

    def add_extension(self, W):
        self.Ws.append((self.stage, self.iteration, W))
        self.iteration += 1

    def add_stage(self):
        self.stage += 1 
        self.iteration = 0

info = None

