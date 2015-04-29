"""
E-M algorithm to detect and separate GMMs
"""

import scipy as sc
import scipy.misc
import scipy.spatial
import scipy.linalg

from scipy import array, eye, ones, log
from scipy.linalg import norm
cdist = scipy.spatial.distance.cdist
multivariate_normal = scipy.random.multivariate_normal
logsumexp = scipy.logaddexp.reduce

from util import closest_permuted_matrix, \
        closest_permuted_vector, column_aerr, column_rerr
from models import GaussianMixtureModel

class EMAlgorithm:
    """The expectation maximisation algorithm. Derivers are expected to
    fill in the expectation and maximisation steps"""
    def __init__( self ):
        pass

    def compute_expectation( self, X, O ):
        """Compute the most likely values of the latent variables; returns lhood"""

        raise NotImplementedError

    def compute_maximisation( self, X, Z, O ):
        """Compute the most likely values of the parameters"""

        raise NotImplementedError

    def run( self, X, O, iter_cb = None, iters=100, eps=1e-5 ):
        """Run with some initial values of parameters O"""

        lhood, Z = self.compute_expectation(X, O)
        for i in xrange( iters ):
            print "Iteration %d, lhood = %f" % (i, lhood)
            O = self.compute_maximisation(X, Z, O)
            # Add error and time to log
            if iter_cb is not None:
                iter_cb( i, O, lhood )

            lhood_, Z = self.compute_expectation(X, O)
            if abs(lhood_ - lhood) < eps and i > 10:
                print "Converged with lhood=%f in %d steps." % ( lhood, i )
                lhood = lhood_
                break
            else:
                lhood = lhood_

        return lhood, Z, O



class GaussianMixtureEM( EMAlgorithm ):
    """
    Gaussian Mixtures EM
    (i) Using k-means++ start
    (ii) Assuming spherical gaussians
    """

    def __init__( self, k, d ):
        self.k, self.d = k, d
        EMAlgorithm.__init__( self )

    def compute_expectation( self, X, O ):
        """Compute the most likely values of the latent variables; returns lhood"""
        _, d = X.shape
        M, sigma, w = O

        total_lhood = 0
        # Get pairwise distances between centers (D_ij = \|X_i - M_j\|)
        D = cdist( X, M.T )
        # Probability dist = 1/2(\sigma^2) D^2 + log w
        Z = - 0.5/sigma**2 * (D**2) + log( w ) - 0.5 * d * log(sigma) # Ignoreing constant term
        total_lhood += logsumexp( logsumexp(Z) )

        # Normalise the probilities (soft EM)
        Z = sc.exp(Z.T - logsumexp(Z, 1)).T
            
        return total_lhood, Z

    def compute_maximisation( self, X, Z, O ):
        """Compute the most likely values of the parameters"""
        N, d = X.shape

        M, sigma, w = O

        # Cluster weights (smoothed)
        # Pseudo counts
        w = Z.sum(axis=0) + 1

        # Get new means
        M = (Z.T.dot( X ).T / w)
        sigma = sc.sqrt(( cdist(X, M.T )**2 * Z).sum()/(d*N))
        w /= w.sum()

        return M, sigma, w

    def kmeanspp_initialisation( self, X ):
        """Initialise means using K-Means++"""
        N, _ = X.shape
        k, d = self.k, self.d
        M = []

        # Choose one center amongst the X at random
        m = sc.random.randint( N )
        M.append( X[m] )

        # Choose k centers
        while( len( M ) < self.k ):
            # Create a probability distribution D^2 from the previous mean
            D = cdist( X, M ).min( 1 )**2
            assert( D.shape == (N,) )

            # Normalise and sample a new point
            D /= D.sum()

            m = sc.random.multinomial( 1, D ).argmax()
            M.append( X[m] )

        M = sc.column_stack( M )
        sigma = cdist( X, M.T ).sum()/(k*d*N)
        w = ones( k )/float(k)

        return M, sigma, w

    def run( self, X, O = None, *args, **kwargs ):
        """O are the 'true' parameters"""
        if O == None:
            O = self.kmeanspp_initialisation( X )
        return EMAlgorithm.run( self, X, O, *args, **kwargs )

def test_gaussian_em():
    """Test the Gaussian EM on a small generated dataset"""
    fname = "gmm-3-10-0.7.npz"
    gmm = GaussianMixtureModel.generate( fname, 3, 3 )
    k, d, M, S, w = gmm.k, gmm.d, gmm.means, gmm.sigmas, gmm.weights
    N, n = 1e6, 1e5


    X = gmm.sample( N, n )

    algo = GaussianMixtureEM(k, d)

    def report( i, O_, lhood ):
        M_, _, _ = O_
    lhood, Z, O_ = algo.run( X, None, report )

    M_, S_, w_ = O_

    M_ = closest_permuted_matrix( M, M_ )
    w_ = closest_permuted_vector( w, w_ )

    print w, w_

    print norm( M - M_ )/norm(M)
    print abs(S - S_).max()
    print norm( w - w_ ) 

    assert( norm( M - M_ )/norm(M) < 1e-1 )
    assert (abs(S - S_) < 1 ).all()
    assert( norm( w - w_ ) < 1e-2 )

def main(fname, N, n, params):
    """Run GMM EM on the data in @fname"""

    gmm = GaussianMixtureModel.from_file( fname )
    k, d, M, S, w = gmm.k, gmm.d, gmm.means, gmm.sigmas, gmm.weights

    X = gmm.sample( N, n )

    # Set seed for the algorithm
    sc.random.seed( int( params.seed ) )

    algo = GaussianMixtureEM( k, d )

    O = M, S, w
    def report( i, O_, lhood ):
        M_, _, _ = O_
    lhood, Z, O_ = algo.run( X, None, report )

    M_, S_, w_ = O_
    M_ = closest_permuted_matrix( M.T, M_.T ).T

    # Table
    print column_aerr( M, M_ ), column_rerr( M, M_ )

if __name__ == "__main__":
    import argparse, time
    parser = argparse.ArgumentParser()
    parser.add_argument( "fname", help="Input file (as npz)" )
    parser.add_argument( "ofname", help="Output file (as npz)" )
    parser.add_argument( "--seed", default=time.time(), type=long, help="Seed used" )
    parser.add_argument( "--samples", type=float, help="Limit number of samples" )
    parser.add_argument( "--subsamples", default=-1, type=float, help="Subset of samples to be used" )

    args = parser.parse_args()

    main( args.fname, int(args.samples), int(args.subsamples), args )

