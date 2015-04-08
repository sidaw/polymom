from cvxopt import matrix, sparse, solvers
import sympy as sp
import numpy as np
import MomentMatrix as mm

def test_unimixture():
    x = sp.symbols('x')
    M = mm.MomentMatrix(3, [x], morder='grevlex')
    constrs = [x-1.5, x**2-2.5, x**4-8.5]
    cin = M.get_cvxopt_inputs(constrs)
    sol = solvers.sdp(cin['c'], Gs=cin['Gs'], \
                  hs=cin['hs'], A=cin['A'], b=cin['b'])

    print sol['x']
    print abs(sol['x'][3]-4.5)
    assert(abs(sol['x'][3]-4.5) <= 1e-5)

def test_2mog(mu=[-1., 4.], sigma=[2., 1.], pi=[0.4, 0.6], deg = 3):
    mu,sig = sp.symbols('mu,sigma')
    M = mm.MomentMatrix(deg, [mu, sig], morder='grevlex')
    constrs = [x-1.5, x**2-2.5, x**4-8.5]
    cin = M.get_cvxopt_inputs(constrs)
    sol = solvers.sdp(cin['c'], Gs=cin['Gs'], \
                  hs=cin['hs'], A=cin['A'], b=cin['b'])

    print sol['x']
    print abs(sol['x'][3]-4.5)
    assert(abs(sol['x'][3]-4.5) <= 1e-5)


test_unimixture()
