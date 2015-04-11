"""
Generate data from a mixture of linear regressions
"""

import scipy as sc
import scipy.linalg
from scipy import array, zeros, ones, eye, rand
from scipy.linalg import inv
from models.Model import Model
from util import chunked_update #, ProgressBar

multinomial = sc.random.multinomial
#multivariate_normal = sc.random.multivariate_normal 
dirichlet = sc.random.dirichlet

from util import permutation, wishart

# import spectral.linalg as sl

class LinearRegressionsMixture( Model ):
    """Generic mixture model with N components"""
    def __init__( self, fname, **params ):
        """Create a mixture model for components using given weights"""
        Model.__init__( self, fname, **params )
        self.k = self.get_parameter( "k" )
        self.d = self.get_parameter( "d" )

        self.weights = self.get_parameter( "w" )
        self.betas = self.get_parameter( "B" )

        self.mean = self.get_parameter( "M" )
        self.sigma = self.get_parameter( "S" )

    @staticmethod
    def from_file( fname ):
        """Load model from a HDF file"""
        model = Model.from_file( fname ) 
        return LinearRegressionsMixture( fname, **model.params )

    def sample( self, N, n = -1 ):
        """Sample N samples from the model. If N, n are both specified,
        then generate N samples, but only keep n of them"""
        if n <= 0: 
            n = N
        shape = (n, self.d)
        y_shape = (n,)

        X = self._allocate_samples( "X", shape )
        y = self._allocate_samples( "y", y_shape )
        # Get a random permutation of N elements
        perm = permutation( N )

        # Sample the number of samples from each view
        cnts = multinomial( N, self.weights )

        mean, sigma = self.mean, self.sigma

        cnt_ = 0
        for i in xrange( self.k ):
            cnt = cnts[i]
            # Generate a bunch of points for each mean
            beta = self.betas.T[i]

            # 1e4 is a decent block size
            def update( start, stop ):
                """Sample random vectors and then assign them to X in
                order"""
                #Z = multivariate_normal( mean, sigma, int(stop - start) )
                Z = 2*rand( int(stop - start), self.d ) - 1
                # Insert into X in a shuffled order
                p = perm[ start:stop ]
                perm_ = p[ p < n ]
                X[ perm_ ] = Z[ p < n ]
                y[ perm_] = Z[ p < n ].dot( beta )
            chunked_update( update, cnt_, 10 ** 4, cnt_ + cnt  )
            cnt_ += cnt
        X.flush()
        y.flush()
        return y, X

    @staticmethod
    def generate( fname, k, d, mean = "zero", cov = "random", betas = "random", weights = "random",
            dirichlet_scale = 10, gaussian_precision = 0.01 ):
        """Generate a mixture of k d-dimensional multi-view gaussians""" 

        model = Model( fname )
        model.add_parameter( "k", k )
        model.add_parameter( "d", d )

        if weights == "random":
            w = dirichlet( ones(k) * dirichlet_scale ) 
        elif weights == "uniform":
            w = ones(k)/k
        elif isinstance( weights, sc.ndarray ):
            w = weights
        else:
            raise NotImplementedError

        if betas == "eye":
            B = sc.eye(d)[:,:k]
        elif betas == "random":
            B = sc.randn( d, k )
        elif isinstance( betas, sc.ndarray ):
            B = betas
        else:
            raise NotImplementedError

        if mean == "zero":
            M = zeros( d )
        elif mean == "random":
            M = sc.randn( d )
        elif isinstance( mean, sc.ndarray ):
            M = mean
        else:
            raise NotImplementedError

        if cov == "eye":
            S = eye( d )
        elif cov == "spherical":
            # Using 1/gamma instead of inv_gamma
            sigma = 1/sc.random.gamma(1/gaussian_precision)
            S = sigma * eye( d )
        elif cov == "random":
            S = gaussian_precision * inv( wishart( d+1, sc.eye( d ), 1 ) ) 
        elif isinstance( cov, sc.ndarray ):
            S = cov
        else:
            raise NotImplementedError

        model.add_parameter( "w", w )
        model.add_parameter( "B", B )
        model.add_parameter( "M", M )
        model.add_parameter( "S", S )

        # Unwrap the store and put it into the appropriate model
        return LinearRegressionsMixture( model.fname, **model.params )

def main( fname, K, d, args ):
    model = LinearRegressionsMixture.generate(fname, K, d, 
                mean = args.mean, cov = args.cov, betas = args.betas, 
                weights = args.weights, dirichlet_scale = args.dirichlet_scale, 
                gaussian_precision = args.gaussian_precision )
    model.set_seed( int( args.seed ) )
    model.save()

if __name__ == "__main__":
    import argparse
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument( "fname", help="Output file (as npz)" )
    parser.add_argument( "k", type=int, help="Number of mixture components"  )
    parser.add_argument( "d", type=int, help="Dimensionality of each component"  )

    parser.add_argument( "--seed", default=int(time.time() * 1000), type=int )

    parser.add_argument( "--betas", default="eye", help="Regression coefficients, default=random|eye" )
    parser.add_argument( "--weights", default="random", help="Mixture weights, default=uniform|random" )
    parser.add_argument( "--dirichlet_scale", default=10.0, type=float, help="Scale parameter for the Dirichlet" )

    parser.add_argument( "--mean", default="zero", help="Mean generation procedure, default = zero" )
    parser.add_argument( "--cov", default="eye", help="Covariance generation procedure, default = spherical|random|eye"  )
    parser.add_argument( "--gaussian_precision", default=0.2, type=float )

    args = parser.parse_args()
    sc.random.seed( int( args.seed ) )

    main( args.fname, args.k, args.d, args )

