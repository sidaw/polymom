#!/usr/bin/env python2.7
"""
Simple tests to verify that polynomial optimization works
"""

import numpy as np
from cvxopt import matrix, solvers
import polyopt
import sympy

def test_quadratic():
    """
    Test the simple quadratic (x-a)^2.
    """

    a = 2.

    x = sympy.symbols('x')
    pol = sympy.poly( (x - a)**2 )
    x_opt = np.array([[1., a, a**2]]).T

    sol = polyopt.optimize_polynomial(pol)
    assert sol['status'] == 'optimal'
    x_sol = np.array( sol['x'] )
    assert np.allclose( x_opt, x_sol, 1e-3 )

def test_symmetric_quadratic():
    """
    x = 1, y = 2
    Test the simple quadratic (x + y - 3)^2 + (x^2 + y^2 - 5)^2.
    """

    x, y = sympy.symbols('x,y')
    a = 2
    pol = sympy.poly( (x + y - 3)**2 +  (x**2 + y**2 - 5)**2 )
    x_opt = np.array([[1., a, a**2]]).T

    sol = polyopt.optimize_polynomial(pol)
    assert sol['status'] == 'optimal'
    x_sol = np.array( sol['x'] )
    print x_sol
    assert np.allclose( x_opt, x_sol, 1e-3 )


