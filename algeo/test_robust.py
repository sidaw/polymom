#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Test how robust border bases are.
"""

import sympy as sp
import numpy as np
from sympy import ring, RR, lex, grlex, grevlex, pprint, Eq, Or
from numpy import sin, cos, pi
from numpy.random import randn

import BorderBasis as BB

import matplotlib.pyplot as plt

def generate_lines(N=2, sigma = 0.):
    R, x, y = ring('x, y', RR, order=grevlex)
    thetas = [2 * pi / N * t for t in xrange(N)]
    I = [x * sin(t) + y * cos(t) + sigma * randn() for t in thetas]

    return R, I

def generate_parabolas(N=2, sigma = 0.):
    R, x, y = ring('x, y', RR, order=grevlex)
    thetas = [2 * pi / N * t for t in xrange(N)]
    I = [(x * sin(t) + y * cos(t) - 1)**2 - (x+cos(t))**2 - (y+sin(t))**2 + sigma * randn() for t in thetas]

    return R, I

def sample_points(f, range_x=(-1.,1.), resolution=100):
    f = sp.poly(f.as_expr())
    pts = []
    for x in np.linspace(*(range_x + (resolution,))):
        for y in sp.polys.real_roots(f(x)):
            pts.append((float(x), float(y)))
    return np.array(pts)

def plot_curves(R, I, range_x=(-1.,1.), resolution=100):
    for i, f in enumerate(I):
        pts = sample_points(f, range_x, resolution)
        plt.scatter(pts.T[0], pts.T[1], s=30, c=plt.cm.rainbow(i * 256 / len(I)), alpha=0.5)
    plt.xlim((-1,1))
    plt.ylim((-1,1))


def do_command(args):
    if args.type == "line":
        R, I = generate_lines(args.n, args.sigma)
    elif args.type == "parabola":
        R, I = generate_parabolas(args.n, args.sigma)
    # Plot the figures
    plot_curves(R, I)

    plt.text(0.1, 0.9, 'sigma = %.3g'%(args.sigma), transform=plt.gca().transAxes)
    for width in [1e-10, 1e-3, 1e-2, 1e-1, 5e-1]:
        try:
            print "trying with width", width
            V = np.array(BB.BorderBasisFactory(width).generate(R,I).zeros())
            # Plot the points
            plt.scatter(V.T[0], V.T[1], c='B', s=(10*width+1)*100, alpha=1.0, marker='*')
            plt.text(0.1, 0.1, 'eps = %.3g'%(width), transform=plt.gca().transAxes)

            break
        except AssertionError:
            continue
    plt.savefig('%s-%d-%.3f.png'%(args.type, args.n, args.sigma))
    #plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser( description='' )
    parser.add_argument('-n', type=int, default=5, help="Number of curves" )
    parser.add_argument('--type', choices=['line', 'parabola'], default='parabola', help="type of curve" )
    parser.add_argument('--sigma', type=float, default=0., help="noise to add" )
    parser.set_defaults(func=do_command)

    #subparsers = parser.add_subparsers()
    #command_parser = subparsers.add_parser('command', help='' )
    #command_parser.set_defaults(func=do_command)

    ARGS = parser.parse_args()
    ARGS.func(ARGS)

