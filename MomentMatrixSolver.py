import sympy as sp
import numpy as np

import sympy.polys.monomials as mn
from sympy.polys.orderings import monomial_key

import scipy.linalg # for schur decomp, which np doesnt have
# import numpy.linalg # for its norm, which suits us better than scipy

from collections import defaultdict
import util
import ipdb
from cvxopt import matrix, sparse, spmatrix, solvers

EPS = 1e-7

def joint_alternating_solver(M, constraints, rank=2, maxiter=100, tol=1e-3):
    lenys = len(M.matrix_monos)
    rowsM = len(M.row_monos)
    lenLs = rank*rowsM
    L = np.random.randn(rank, len(M.row_monos))
    L = np.array([[1,1,1,1],[1,2,4,8]])/np.sqrt(2)
    Bf = M.get_Bflat()
    
    A,b = M.get_Ab(constraints, cvxoptmode = False)

    weightone = 1
    A = A*weightone; b = b*weightone
    
    consts = np.zeros((lenLs + lenys, 1))
    consts[-lenys:] = A.T.dot(b)
    coeffs = np.zeros((lenLs+lenys,lenLs+lenys))

    for i in xrange(maxiter):
        smallblock = L.dot(L.T)
        dy_L = -Bf.dot(np.kron(np.eye(rowsM), L.T))
        dy_y = A.T.dot(A) + Bf.dot(Bf.T)

        dL_L = np.kron(np.eye(rowsM), smallblock)
        dL_y = -dy_L.T

        coeffs[0:lenLs, 0:lenLs] = dL_L
        coeffs[0:lenLs, lenLs:] = dL_y

        coeffs[lenLs:, 0:lenLs] = dy_L
        coeffs[lenLs:, lenLs:] = dy_y
        
        sol = scipy.linalg.solve(coeffs, consts)
        
        L = sol[0:lenLs].reshape((rank, rowsM))

        # hack to normalize
        # L = L / scipy.linalg.norm(L[:,0])
        y = sol[-lenys:]
        obj = scipy.linalg.norm(L.T.dot(L) - M.numeric_instance(y)) + scipy.linalg.norm(A.dot(y)-b)
        print obj
    ipdb.set_trace()
    # need to know which matrix_mono is in each location

def sgd_solver(M, constraints, rank=2, maxiter=100, eta = 1):
    lenys = len(M.matrix_monos)
    rowsM = len(M.row_monos)
    lenLs = rank*rowsM
    L = np.random.randn(rank, len(M.row_monos))
    L = np.array([[1,1,1,1],[1,2,4,8]])/np.sqrt(2)
    y = np.random.randn(lenys,1)
    Bf = M.get_Bflat()
    
    A,b = M.get_Ab(constraints, cvxoptmode = False)

    weightone = 1
    A = A*weightone; b = b*weightone
    
    consts = np.zeros((lenLs + lenys, 1))
    consts[-lenys:] = A.T.dot(b)
    coeffs = np.zeros((lenLs+lenys,lenLs+lenys))

    Ly = np.zeros((lenLs + lenys,1))
    Ly[0:lenLs] = L.T.flatten()[:,np.newaxis]
    Ly[lenLs:] = np.random.randn(lenys, 1)

    for i in xrange(maxiter):
        smallblock = L.dot(L.T)
        dy_L = -Bf.dot(np.kron(np.eye(rowsM), L.T))
        dy_y = A.T.dot(A) + Bf.dot(Bf.T)

        dL_L = np.kron(np.eye(rowsM), smallblock)
        dL_y = -dy_L.T

        coeffs[0:lenLs, 0:lenLs] = dL_L
        coeffs[0:lenLs, lenLs:] = dL_y

        coeffs[lenLs:, 0:lenLs] = dy_L
        coeffs[lenLs:, lenLs:] = dy_y
        
        grad = coeffs.dot(Ly) - consts;
        Ly = Ly - eta * grad
        L = Ly[0:lenLs].reshape((rowsM, rank)).T
        
        obj = scipy.linalg.norm(L.T.dot(L) - M.numeric_instance(y))**2 + scipy.linalg.norm(A.dot(y)-b)**2
        print obj
    ipdb.set_trace()
    # need to know which matrix_mono is in each location

def alternating_sgd_solver(M, constraints, rank=2, maxiter=100, tol=1e-3, eta = 1e-2):
    lenys = len(M.matrix_monos)
    rowsM = len(M.row_monos)
    lenLs = rank*rowsM
    La = np.random.randn(rank, len(M.row_monos))
    #La = np.array([[1,1,1,1],[1,2,4,8]])/np.sqrt(2)
    
    coeffs = np.zeros((lenLs+lenys, lenLs+lenys))
    consts = np.zeros((lenLs+lenys, 1))
    
    A,b = M.get_Ab(constraints, cvxoptmode = False)
    
    b = -A[:-1,0][:,np.newaxis]
    A = A[:-1,1:]
    
    counts = [len(M.term_to_indices_dict[yi]) for yi in M.matrix_monos[1:]]
    weight_fit = 1;
    for i in xrange(maxiter):
        # update y
        Q_y = A.T.dot(A) + weight_fit*np.diag(counts)
        currentM = La.T.dot(La)
        weights = [sum(currentM.flatten()[M.term_to_indices_dict[yi]]) for yi in M.matrix_monos[1:]]
        p_y = A.T.dot(b) + weight_fit*np.array(weights)[:,np.newaxis]
        y,_,_,_ = scipy.linalg.lstsq(Q_y, p_y)
        y_one = np.vstack((1,y))
        # print y, La.T.dot(La)
        # update L
        #ipdb.set_trace()
        Q_l = La.dot(La.T)
        p_l = La.dot(M.numeric_instance( y_one ))
        grad = Q_l.dot(La) - p_l
        #La,_,_,_ = scipy.linalg.lstsq(Q_l, p_l)
        La = La - eta*grad
        if i % 50 == 0:
            obj = scipy.linalg.norm(La.T.dot(La) - M.numeric_instance(y_one))**2 + scipy.linalg.norm(A.dot(y)-b)**2
            print i,obj

    return y,La

def alternating_solver(M, constraints, rank=2, maxiter=100, tol=1e-3):
    lenys = len(M.matrix_monos)
    rowsM = len(M.row_monos)
    lenLs = rank*rowsM
    La = np.random.randn(rank, len(M.row_monos))
    #La = np.array([[1,1,1,1],[1,2,4,8]])/np.sqrt(2)
    
    coeffs = np.zeros((lenLs+lenys, lenLs+lenys))
    consts = np.zeros((lenLs+lenys, 1))
    
    A,b = M.get_Ab(constraints, cvxoptmode = False)
    
    b = -A[:-1,0][:,np.newaxis]
    A = A[:-1,1:]
    
    counts = [len(M.term_to_indices_dict[yi]) for yi in M.matrix_monos[1:]]
    weight_fit = 1;
    for i in xrange(maxiter):
        # update y
        Q_y = A.T.dot(A) + weight_fit*np.diag(counts)
        currentM = La.T.dot(La)
        weights = [sum(currentM.flatten()[M.term_to_indices_dict[yi]]) for yi in M.matrix_monos[1:]]
        p_y = A.T.dot(b) + weight_fit*np.array(weights)[:,np.newaxis]
        y,_,_,_ = scipy.linalg.lstsq(Q_y, p_y)
        y_one = np.vstack((1,y))
        # print y, La.T.dot(La)
        # update L
        #ipdb.set_trace()
        Q_l = La.dot(La.T)
        p_l = La.dot(M.numeric_instance( y_one ))
        La,_,_,_ = scipy.linalg.lstsq(Q_l, p_l)
        
        if i % 50 == 0:
            obj = scipy.linalg.norm(La.T.dot(La) - M.numeric_instance(y_one))**2 + scipy.linalg.norm(A.dot(y)-b)**2
            print i,obj

    return y,La
    # need to know which matrix_mono is in each location
