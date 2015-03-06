#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_hists(*vs, **kwargs):
    """
    Plot a histogram jointly for each graph in vs
    """
    n_bins = kwargs.get('n_bins', 100)
    labels = kwargs.get('labels', range(len(vs)))

    x_min = min(map(min, vs))
    x_max = max(map(max, vs))

    bins = np.linspace(x_min, x_max, n_bins)
    for v, lbl in zip(vs, labels):
        plt.hist(v, bins, alpha=0.5, label=str(lbl))
    plt.legend(loc='upper right')

