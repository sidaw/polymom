"""
Tests for mixture models
"""

import numpy as np
import polymom

SIMPLE_HYPER = {
        'k' : 2,
        'd' : 2,
        }
SIMPLE_PARAMS = {
        (1,) : 0.4,
        (2,) : 0.6,
        (1,1) : 1,
        (1,2) : 0,
        (2,1) : 0,
        (2,2) : 1,
        }

def test_param_generation():
    """Test that randomly generated polynomials are correct"""
    for k, d in [(2,2), (2,3), (3,3)]:
        params = polymom.generate_parameters({'k' : k, 'd' : d})
        assert len(params) == k + k*d
        assert np.allclose(sum( params[(h,)] for h in xrange(1, k+1) ), 1.0)
        for h in xrange(1,k+1):
            assert np.allclose(sum( params[(h,i)] for i in xrange(1, d+1) ), 1.0)

def test_poly_generation():
    """
    Generate a polynomial from parameters
    """
    poly = polymom.generate_poly( SIMPLE_HYPER, SIMPLE_PARAMS )
    print poly

def test_simple_mom_recovery():
    pass


