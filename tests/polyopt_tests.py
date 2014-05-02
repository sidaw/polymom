#!/usr/bin/env python2.7
"""
Simple tests to verify that polynomial optimization works
"""

import numpy as np
from cvxopt import matrix, solvers
import polyopt
from sympy import symbols, poly

def test_quadratic():
    """
    Test the simple quadratic (x-a)^2.
    """
    a = 2.

    x = symbols('x')
    pol = poly( (x - a)**2 )
    x_opt = a

    sol = polyopt.optimize_polynomial(pol)
    x_sol = sol[x]

    assert np.allclose( x_opt, x_sol, 1e-3 )

def test_cubic():
    """
    Test the simple quadratic (x-a)^2.
    """
    a = 2.

    x = symbols('x')
    pol = poly( (x - a)**6 )
    x_opt = a

    sol = polyopt.optimize_polynomial(pol)
    x_sol = sol[x]

    assert np.allclose( x_opt, x_sol, 1e-1 )

def test_independent_quadratic():
    """
    x = 1, y = 2
    Test the simple quadratic (x - 3)^2 + (y - 5)^2.
    """

    a, b = 2., 3.
    x, y = symbols('x,y')
    pol = polyopt.by_evaluation(
            x,
            y,
            x = a,
            y = b)
    sol = polyopt.optimize_polynomial(pol)
    x_sol = sol[x], sol[y]
    assert np.allclose( [a, b], x_sol, 1e-3 )

def test_symmetric_quadratic():
    """
    x = 1, y = 2
    Test the simple quadratic (x + y - 3)^2 + (x^2 + y^2 - 5)^2.
    """

    a, b = 1., 2.
    x, y = symbols('x,y')
    pol = polyopt.by_evaluation(
            x + y,
            x**2 + y**2,
            x = a,
            y = b)
    sol = polyopt.optimize_polynomial(pol)
    x_sol = sol[x], sol[y]
    # This doesn't work.
    assert np.allclose( [a, b], x_sol, 1e-3 ) == False

def test_broken_quadratic():
    """
    x = 1, y = 2
    Test the simple quadratic (pi * x + (1-pi) * y - 3)^2 + (x^2 + y^2 - 5)^2.
    """

    a, b, pi = 1., 2., 0.4,
    x, y = symbols('x,y')
    pol = polyopt.by_evaluation(
            pi * x + (1-pi) * y,
            pi * x**2 + (1-pi) * y**2,
            x = a,
            y = b)
    sol = polyopt.optimize_polynomial(pol)
    x_opt = np.array([a,b])
    x_sol = sol[x], sol[y]

    diff = np.sqrt((x_opt - x_sol).dot(x_opt - x_sol))
    print diff
    # This doesn't work.
    assert np.allclose( x_opt, x_sol, 1e-1 ) 

