#!/usr/bin/env python2.7
"""
Simple tests to verify that polynomial optimization works
"""

import numpy as np
from cvxopt import matrix, solvers

def test_cvx_simple():
    """
    Test that CVX works on a simple LP.
    In particular:
    min             2x_1 + x_2              (c'x)
    subject to      -x_1 + x_2  <= 1        (Ax <= b)
                     x_1 + x_2  >= 2
                     x_2        >= 0
                     x_1 - 2x_2 <= 4
    Taken from CVXOPT website: http://cvxopt.org/examples/tutorial/lp.html
    """
    A = matrix([ [-1.0, -1.0, 0.0, 1.0], [1.0, -1.0, -1.0, -2.0] ])
    b = matrix([1.0, -2.0, 0.0, 4.0])
    c = matrix([2.0, 1.0])
    sol = solvers.lp(c, A, b)

    x = np.matrix(sol['x'])
    x_ = np.matrix([0.5, 1.5]).T
    print x, x_
    assert np.allclose(x, x_)

def test_cvx_sdp():
    """
    Test that the CVX SDP optimization works.
    """
    c = matrix([1., -1., 1.])
    G = [ matrix([[-7., -11., -11., 3.],
                  [ 7., -18., -18., 8.],
                  [-2.,  -8.,  -8., 1.]]) ]
    G += [ matrix([[-21., -11.,   0., -11.,  10.,   8.,   0.,   8., 5.],
                   [  0.,  10.,  16.,  10., -10., -10.,  16., -10., 3.],
                   [ -5.,   2., -17.,   2.,  -6.,   8., -17.,  8., 6.]]) ]
    h = [ matrix([[33., -9.], [-9., 26.]]) ]
    h += [ matrix([[14., 9., 40.], [9., 91., 10.], [40., 10., 15.]]) ]
    sol = solvers.sdp(c, Gs=G, hs=h)
    x = np.matrix(sol['x'])
    x_ = np.matrix([
        [-3.68e-01],
        [ 1.90e+00],
        [-8.88e-01],
        ])
    print x, x_
    assert np.allclose(x, x_, atol=1e-2)

def test_cvx_polynomial():
    """
    Test that the SDP relaxation for polynomial optimization works on simple examples. 
    In particular, consider min (x-2)^2.
    We get this to be x^2 - 2x + 1, which is
    min [1 -2 1]' y, 
    subj to yy' >= 0
    """

    c = matrix([1.0, -2, 1.0])
    # No equality constraint
    # This is painful

    assert False


