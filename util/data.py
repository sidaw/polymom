"""
Data manipulation functions
"""

import numpy as np

#from . import _data

def Triples(x1, x2, x3):
    """Compute E[x1 \ctimes x2 \ctimes x3 ]"""

    N = x1.shape[0]
    d = x1.shape[1]
    triples = np.zeros( (d,d,d))

    # Compute one element of Triples at a time
    for n in range( N ):
        for k in range( d ):
            for j in range( d ):
                for i in range( d ):
                    triples[i,j,k] += (x1[n,i] * x2[n,j] * x3[n,k] - triples[i,j,k])/(n+1)
    return triples

#def count_frequency( X, d ):
#    N, W = X.shape
#    Y = _data.count_frequency( X, d )
#    return Y/float(W)

#Pairs = _data.Pairs
#Pairs2 = _data.Pairs2
#PairsQ = _data.PairsQ
#Triples = _data.Triples
#TriplesQ = _data.TriplesQ
#TriplesP = _data.TriplesP
#TriplesPQ = _data.TriplesPQ
#
#xMy = _data.xMy
#Txyz = _data.Txyz
#
#def test_pairs():
#    """Test Pairs"""
#    
#    # Generate N random vectors
#    N = 1000
#    d = 10
#
#    X1 = sc.random.rand( N, d )
#    X2 = sc.random.rand( N, d )
#
#    P = Pairs( X1, X2 )
#
#    P_ = sc.zeros( (d,d) )
#    for i in xrange( N ):
#        P_ += (sc.outer(X1[i], X2[i]) - P_)/(i+1)
#
#    assert sc.allclose( P, P_ )
#
#def test_triplesp():
#    """Test Pairs"""
#    
#    # Generate N random vectors
#    N = 1000
#    d = 10
#
#    X1 = sc.random.rand( N, d )
#    X2 = sc.random.rand( N, d )
#    X3 = sc.random.rand( N, d )
#
#    theta = sc.random.rand( d )
#
#    T = TriplesP( X1, X2, X3, theta )
#
#    T_ = sc.zeros( (d,d) )
#    for i in xrange( N ):
#        T_ += (sc.outer(X1[i], X2[i]) * theta.dot(X3[i]) - T_)/(i+1)
#
#    assert sc.allclose( T, T_ )
#
