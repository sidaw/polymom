#!/usr/bin/env python2.7
"""
Simple tests to verify that polynomial optimization works
"""

import numpy as np
import cvxpy as cp
import cvxopt

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

    x_ = np.matrix([0.5, 1.5]).T

    x = cp.Variable(2)
    c = np.matrix([[2, 1]]).T
    A = np.matrix([[-1, 1],
                  [ -1, -1],
                  [ 0, -1],
                  [ 1, -2],
                  ])
    b = np.matrix([[ 1, -2, 0, 4 ]]).T

    prob = cp.Problem(
            cp.Minimize( c.T * x ),
            [A * x - b <= 0],
            )
    prob.solve()

    print prob.status
    x = np.matrix(x.value)
    print 'x', x
    print 'x*', x_
    assert np.allclose(x, x_)

def test_cvx_sdp():
    """
    Test that the CVX SDP optimization works.
    """
    x1, x2, x3 = cp.Variable(), cp.Variable(), cp.Variable(),

    A1 = np.matrix([[-7, -11], 
                    [-11, 3]])
    A2 = np.matrix([[7, -18], 
                    [-18, 8]])
    A3 = np.matrix([[-2, -8], 
                    [-8, 1]])
    A0 = np.matrix([[33, -9], 
                    [-9, 26]])

    B1 = np.matrix([[-21, -11, 0], 
                    [-11, 10,  8],
                    [0, 8,  5], ])
    B2 = np.matrix([[ 0,  10, 16], 
                    [10, -10, -10],
                    [16, -10,  3], ])
    B3 = np.matrix([[-5, 2, -17], 
                    [2, -6,  -7],
                    [-17, 8,  6], ])
    B0 = np.matrix([[14, 9, 40], 
                    [9, 91,  10],
                    [40, 10,  15], ])


    prob = cp.Problem(
        cp.Minimize( x1 - x2 + x3 ), [
            A0 - x1 * A1 - x2 * A2 - x3 * A3 == cp.semidefinite(2),
            B0 - x1 * B1 - x2 * B2 - x3 * B3 == cp.semidefinite(3),
            ])

    prob.solve()

    print prob.status

    x = np.matrix([
        x1.value,
        x2.value,
        x3.value,
        ]).T
    x_ = np.matrix([
        [-3.68e-01],
        [ 1.90e+00],
        [-8.88e-01],
        ])
    print x.T, x_.T
    assert np.allclose(x, x_, atol=1e-2)

def test_cvx_polynomial():
    """
    Test that the SDP relaxation for polynomial optimization works on simple examples. 
    In particular, consider min (x-2)^2.
    We get this to be x^2 - 2x + 1, which is
    min [1 -2 1]' y, 
    subj to yy' >= 0
    """

    a = 1.0

    A = cp.Variable(2,2)

    # No equality constraint

    prob = cp.Problem( 
        cp.Minimize( a**2 * A[0,0] - 2 * a * A[0,1] + A[1,1] ), [
             A == cp.semidefinite(2),
             A[0,0] == 1.
            ])
    prob.solve()

    print prob.status
    print A.value

    assert False


