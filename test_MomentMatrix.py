from cvxopt import matrix, sparse, solvers
import sympy as sp
import numpy as np
import MomentMatrix as mm

def test_unimixture():
    print 'testing simple unimixture with a skipped observation'
    x = sp.symbols('x')
    M = mm.MomentMatrix(3, [x], morder='grevlex')
    constrs = [x-1.5, x**2-2.5, x**4-8.5]
    cin = M.get_cvxopt_inputs(constrs)
    sol = solvers.sdp(cin['c'], Gs=cin['Gs'], \
                  hs=cin['hs'], A=cin['A'], b=cin['b'])

    print sol['x']
    print abs(sol['x'][3]-4.5)
    assert(abs(sol['x'][3]-4.5) <= 1e-5)

# helper function to generate coeffs of the Gaussian moments
# they are non-neg and equal in abs to the coeffs hermite polynomials
def hermite_coeffs(N=6):
    K = N
    A = np.zeros((N,K), dtype=np.int)
    # the recurrence formula to get coefficients of the hermite polynomails
    A[0,0] = 1; A[1,1] = 1; #A[2,0]=-1; A[2,2]=1;
    for n in range(1,N-1):
        for k in range(K):
            A[n+1,k] = -n*A[n-1,k] if k==0 else A[n,k-1] - n*A[n-1,k]
    return A
            

def test_1dmog(mus=[-1., 4.], sigs=[1., 1.], pi=[0.5, 0.5], deg = 4):
    print 'testing 1d mixture of Gaussians'
    K = len(mus)
    mu,sig = sp.symbols('mu,sigma')
    M = mm.MomentMatrix(deg, [mu, sig], morder='grevlex')

    num_constrs = 9; # so observe num_constrs-1 moments
    H = abs(hermite_coeffs(num_constrs))
    constrs = [0]*num_constrs
    
    for order in range(num_constrs):
        for i in range(order+1):
            constrs[order] = constrs[order] + H[order,i]* mu**(i) * sig**(order-i)
        constrsval = 0;
        for k in range(K):
            constrsval += pi[k]*constrs[order].evalf(subs={mu:mus[k], sig:sigs[k]})
        constrs[order] -= constrsval
    print constrs
        
    cin = M.get_cvxopt_inputs(constrs[1:])
    sol = solvers.sdp(cin['c'], Gs=cin['Gs'], \
                  hs=cin['hs'], A=cin['A'], b=cin['b'])
    #import pdb; pdb.set_trace()
    print M.matrix_monos
    print sol['x']

test_unimixture()
test_1dmog()
