"""
Generate data from a Gaussian mixture model
"""

import numpy as np
import scipy as sc
import scipy.linalg
from scipy import array, zeros, ones, eye
from scipy.linalg import inv
from models.Model import Model
from util import chunked_update #, ProgressBar

from sktensor import ktensor

multinomial = sc.random.multinomial
multivariate_normal = sc.random.multivariate_normal 
dirichlet = sc.random.dirichlet

from util import permutation, wishart
from util import hermite_coeffs, tensorify
import sympy as sp

from itertools import combinations_with_replacement, combinations
from collections import Counter
# import spectral.linalg as sl

class GaussianMixtureModel( Model ):
    """Generic mixture model with N components"""
    def __init__( self, fname, **params ):
        """Create a mixture model for components using given weights"""
        Model.__init__( self, fname, **params )
        self.k = self.get_parameter( "k" )
        self.d = self.get_parameter( "d" )
        self.weights = self.get_parameter( "w" )
        self.means = self.get_parameter( "M" )
        self.sigmas = self.get_parameter( "S" )

        # symbolica means and covs
        self.sym_means = sp.symbols('xi1:'+str(self.d+1))
        self.sym_covs =  sp.symbols('c1:'+str(self.d+1))

    @staticmethod
    def from_file( fname ):
        """Load model from a HDF file"""
        model = Model.from_file( fname ) 
        return GaussianMixtureModel( fname, **model.params )

    def sample( self, N, n = -1 ):
        """Sample N samples from the model. If N, n are both specified,
        then generate N samples, but only keep n of them"""
        if n <= 0: 
            n = N
        shape = (n, self.d)

        X = self._allocate_samples( "X", shape )
        # Get a random permutation of N elements
        perm = permutation( N )

        # Sample the number of samples from each view
        cnts = multinomial( N, self.weights )

        cnt_ = 0
        for i in xrange( self.k ):
            cnt = cnts[i]
            # Generate a bunch of points for each mean
            mean, sigma = self.means.T[ i ], self.sigmas[ i ]

            # 1e4 is a decent block size
            def update( start, stop ):
                """Sample random vectors and then assign them to X in
                order"""
                Y = multivariate_normal( mean, sigma, int(stop - start) )
                # Insert into X in a shuffled order
                p = perm[ start:stop ]
                perm_ = p[ p < n ]
                X[ perm_ ] = Y[ p < n ]
            chunked_update( update, cnt_, 10 ** 4, cnt_ + cnt  )
            cnt_ += cnt
        X.flush()
        return X

    def get_exact_moments(self):
        d = self.d
        sigma2 = self.sigmas[0,0,0]
        M1 = sum( w * mu for w, mu in zip(self.weights, self.means.T)).reshape(d,1)
        M2 = sum( w * (sc.outer(mu,mu) + S) for w, mu, S in zip(self.weights, self.means.T, self.sigmas))
        M3 = sum( w * tensorify(mu,mu,mu) for w, mu, S in zip(self.weights, self.means.T, self.sigmas))

        M1_ = np.hstack( (M1 for _ in range(d)) )
        M3 += sigma2 * ktensor([M1_, np.eye(d), np.eye(d)]).totensor()
        M3 += sigma2 * ktensor([np.eye(d), M1_, np.eye(d)]).totensor()
        M3 += sigma2 * ktensor([np.eye(d), np.eye(d), M1_]).totensor()

        return M1, M2, M3



    @staticmethod
    def polymom_univariate(xi, c, order):
        constr = 0
        H = abs(hermite_coeffs(order+1))
        for i in range(order+1):
            constr = constr + H[order,i]* xi**(i) * c**((order-i)/2)
        return constr

    @staticmethod
    def polymom_diag(xis, covs, dims = (1, 2, 2, 3)):
        # a formula for the general moment is here:
        # https://www.stats.bris.ac.uk/research/stats/reports/2002/0211.pdf
        # but it feels like a programming contest problem, and I'll just do
        # the diagonal version for now
        dimtodeg = Counter()
        for d in dims:
            dimtodeg[d] += 1
        
        expr = 1
        for dim,deg in dimtodeg.items():
            expr = expr * ( GaussianMixtureModel.polymom_univariate(xis[dim-1], covs[dim-1], deg) )
        return expr.expand()

    def polymom_all_expressions(self, maxdeg):
        # xis are the means of the Gaussian
        d = self.d
        xis = self.sym_means
        covs = self.sym_covs
        exprs = []

        for deg in range(1,maxdeg+1):
            for indices in combinations_with_replacement(range(1,d+1),deg):
                exprs.append(GaussianMixtureModel.polymom_diag(xis,covs,indices))
        return exprs

    def polymom_all_constraints(self, maxdeg):
        d = self.d
        xis = self.sym_means
        covs = self.sym_covs

        exprs = self.polymom_all_expressions(maxdeg)
        import mompy as mp
        meas = mp.Measure(xis+covs)

        for i in range(self.k):
            means,covs = self.means.T[ i ], sc.diag(self.sigmas[ i ])
            meas += (self.weights[i], means.tolist() + covs.tolist())
        meas.normalize()
        
        for i,expr in enumerate(exprs):
            exprval = meas.integrate(expr)
            exprs[i] = expr - exprval

        return exprs

    def polymom_all_constraints_samples(self, maxdeg, X):
        # xis are the means of the Gaussian
        d = self.d
        xis = self.sym_means
        covs = self.sym_covs
        exprs = []

        for deg in range(1,maxdeg+1):
            for indices in combinations_with_replacement(range(1,d+1),deg):
                m_hat = sc.mean(sc.prod(X[:, sc.array(indices)-1],1),0)
                exprs.append(GaussianMixtureModel.polymom_diag(xis,covs,indices) - m_hat)
        return exprs

    def polymom_monos(self, deg):
        """ return monomials needed to fit this model
        """
        import sympy.polys.monomials as mn
        allvars = self.sym_means + self.sym_covs
        rawmonos = mn.itermonomials(allvars, deg)
        
        # filter out anything whose total degree in cov is greater than deg
        filtered = []
        for mono in rawmonos:
            pd = mono.as_powers_dict()
            sumcovdeg = sum([2*pd[covvar] for covvar in self.sym_covs])
            sumcovdeg += sum([pd[meanvar] for meanvar in self.sym_means])
            if sumcovdeg <= deg:
                filtered.append(mono)
        return filtered
    
    @staticmethod
    def generate( fname, k, d, means = "hypercube", cov = "spherical",
            weights = "random", dirichlet_scale = 10, gaussian_precision
            = 0.01 ):
        """Generate a mixture of k d-dimensional gaussians""" 

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

        if means == "hypercube":
            # Place means at the vertices of the hypercube
            M = zeros( (d, k) )
            if k <= 2**d:
                # the minimum number of ones needed to fill k of them
                numones = int(sc.ceil(sc.log(k)/sc.log(d)))
                allinds = combinations(range(d), numones)
                for i,inds in enumerate(allinds):
                    if i == k: break
                    M[inds, i] = 1
            else:
                raise NotImplementedError
                
        elif means == "random":
            M = 5*sc.randn( d, k )
        elif isinstance( means, sc.ndarray ):
            M = means
        else:
            raise NotImplementedError

        if cov == "spherical":
            # Using 1/gamma instead of inv_gamma
            sigmas = []
            for i in xrange(k):
                sigmak = 0.5*sc.random.rand()+0.5
                sigmas = sigmas + [ sigmak * eye( d ) ]
            S = array( sigmas )
        elif cov == "spherical_uniform":
            # Using 1/gamma instead of inv_gamma
            sigma = 1/sc.random.gamma(1/gaussian_precision)
            S = array( [ sigma * eye( d ) for i in xrange( k ) ] )
        elif cov == "diagonal":
            # Using 1/gamma instead of inv_gamma
            sigmas = []
            for i in xrange(k):
                sigmak = [3*sc.random.rand()+1 for i in xrange(d)]
                sigmas = sigmas + [ sc.diag(sigmak) ]
            S = array( sigmas )
        elif isinstance( cov, sc.ndarray ):
            S = cov
        elif cov == "random":
            S = array( [ gaussian_precision * inv( wishart( d+1, sc.eye( d ), 1 ) ) for i in xrange( k ) ] )
        else:
            raise NotImplementedError

        model.add_parameter( "w", w )
        model.add_parameter( "M", M )
        model.add_parameter( "S", S )

        # Unwrap the store and put it into the appropriate model
        return GaussianMixtureModel( model.fname, **model.params )

    def get_log_likelihood(self, X):
        lhood = 0.
        for x in X:
            lhood_ = 0.
            for i in xrange(self.k):
                lhood_ += self.weights[i] * sc.exp( -0.5 * (self.means[i] - x).dot(self.sigmas[i]).dot(self.means[i] - x) )
            lhood += sc.log(lhood_)
        return lhood


def test_gaussian_mixture_generator_dimensions():
    "Test the GaussianMixtureModel generator"
    import tempfile
    fname = tempfile.mktemp()

    N = 1000
    D = 10
    K = 3

    gmm = GaussianMixtureModel.generate( fname, K, D )
    assert( gmm.means.shape == (D, K) )
    assert( gmm.weights.shape == (K,) )

    X = gmm.sample( N )
    assert( X.shape == (N, D) )

def test_gaussian_mixture_generator_replicatability():
    "Test the GaussianMixtureModel generator"
    import tempfile
    fname = tempfile.mktemp()

    N = 1000
    n = 500
    D = 10
    K = 3

    gmm = GaussianMixtureModel.generate( fname, K, D )
    gmm.set_seed( 100 )
    gmm.save()

    X = gmm.sample( N )
    del gmm

    gmm = GaussianMixtureModel.from_file( fname )
    Y = gmm.sample( N )
    assert( sc.allclose( X, Y ) )
    del gmm

    gmm = GaussianMixtureModel.from_file( fname )
    Y = gmm.sample( N, n )
    assert( sc.allclose( X[:n], Y ) )


class MultiViewGaussianMixtureModel( Model ):
    """Generic mixture model with N components"""
    def __init__( self, fname, **params ):
        Model.__init__( self, fname, **params )
        self.k = self.get_parameter( "k" )
        self.d = self.get_parameter( "d" )
        self.n_views = self.get_parameter( "v" )
        self.weights = self.get_parameter( "w" )
        self.means = self.get_parameter( "M" )
        self.sigmas = self.get_parameter( "S" )

    @staticmethod
    def from_file( fname ):
        """Load model from a HDF file"""
        model = Model.from_file( fname ) 
        return MultiViewGaussianMixtureModel( fname, **model.params )

    def sample( self, N, n = -1 ):
        """Sample n samples from the mixture model"""
        if n <= 0: 
            n = N

        shape = (n, self.d)

        X = []
        for i in xrange( self.n_views ):
            X.append( self._allocate_samples( "X%d" % (i+1), shape ) )
        # Get a random permutation of N elements
        perm = permutation( N )

        # Sample the number of samples from each view
        cnts = multinomial( N, self.weights )

        # Data for each view
        for view in xrange(self.n_views):
            cnt_ = 0
            for i in xrange( self.k ):
                cnt = cnts[i]
                # Generate a bunch of points for each mean
                mean, sigma = self.means[view].T[ i ], self.sigmas[view][ i ]

                def update( start, stop ):
                    """Sample random vectors and then assign them to X in
                    order"""
                    Y = multivariate_normal( mean, sigma, int(stop - start) )
                    # Insert into X in a shuffled order
                    p = perm[ start:stop ]
                    perm_ = p[ p < n ]
                    X[view][ perm_ ] = Y[ p < n ]
                chunked_update( update, cnt_, 10 ** 4, cnt_ + cnt  )
                cnt_ += cnt

        return X

    @staticmethod
    def generate( fname, k, d, n_views = 3, means = "hypercube", cov =
        "spherical", weights = "random", dirichlet_scale = 10,
        gaussian_precision = 0.01 ):
        """Generate a mixture of k d-dimensional multi-view gaussians""" 

        model = Model( fname )
        model.add_parameter( "k", k )
        model.add_parameter( "d", d )
        model.add_parameter( "v", n_views )

        if weights == "random":
            w = dirichlet( ones(k) * dirichlet_scale ) 
        elif weights == "uniform":
            w = ones(k)/k
        elif isinstance( weights, sc.ndarray ):
            w = weights
        else:
            raise NotImplementedError

        if means == "hypercube":
            # Place means at the vertices of the hypercube
            M = []
            for i in xrange( n_views ):
                m = zeros( (d, k) )
                for j in xrange(k):
                    m[(i+j) % k, (i+j) % k] = 1.0
                M.append( m )
            M = array( M )
        elif means == "random":
            M = []
            for i in xrange( n_views ):
                M.append( sc.randn( d, k ) )
            M = array( M )
        elif isinstance( means, sc.ndarray ):
            M = means
        else:
            raise NotImplementedError

        if cov == "spherical":
            # Using 1/gamma instead of inv_gamma
            S = []
            for i in xrange( n_views ):
                sigma = 1/sc.random.gamma(1/gaussian_precision)
                s = array( [ sigma * eye( d ) for i in xrange( k ) ] )
                S.append( s )
            S = array( S ) 
        elif isinstance( cov, sc.ndarray ):
            S = cov
        elif cov == "random":
            S = []
            for i in xrange( n_views ):
                # Roughly the largest element if p = 2d ~= 1, so well
                # scaled.
                s = array( [ gaussian_precision * inv( wishart( d+1, sc.eye( d ), 1 ) ) for i in xrange( k ) ] )
                S.append( s )
            S = array( S ) 
        else:
            raise NotImplementedError

        model.add_parameter( "w", w )
        model.add_parameter( "M", M )
        model.add_parameter( "S", S )

        # Unwrap the store and put it into the appropriate model
        return MultiViewGaussianMixtureModel( model.fname, **model.params )

def test_mv_gaussian_mixture_generator_dimensions():
    "Test the MultiViewGaussianMixtureModel generator"
    import tempfile
    fname = tempfile.mktemp()

    N = 1000
    D = 10
    K = 3
    VIEWS = 3

    mvgmm = MultiViewGaussianMixtureModel.generate( fname, K, D, n_views = VIEWS )
    assert( len( mvgmm.means ) == K )
    assert( len( mvgmm.sigmas ) == K )
    for view in xrange(VIEWS):
        assert( mvgmm.means[view].shape == (D, K) )
    assert( mvgmm.weights.shape == (K,) )
    points = mvgmm.sample( N )
    # 1000 points with 100 dim each
    assert( len( points ) == K )
    for view in xrange(VIEWS):
        assert( points[view].shape == (N, D) )

def test_mv_gaussian_mixture_generator_replicatability():
    "Test the GaussianMixtureModel generator"
    import tempfile
    fname = tempfile.mktemp()

    N = 1000
    n = 500
    D = 10
    K = 3
    VIEWS = 3

    mvgmm = MultiViewGaussianMixtureModel.generate( fname, K, D, n_views = VIEWS )
    mvgmm.set_seed( 100 )
    mvgmm.save()

    X = mvgmm.sample( N )
    del mvgmm

    mvgmm = MultiViewGaussianMixtureModel.from_file( fname )
    Y = mvgmm.sample( N )
    for v in xrange( VIEWS ):
        assert( sc.allclose( X[v], Y[v] ) )
    del mvgmm

    mvgmm = MultiViewGaussianMixtureModel.from_file( fname )
    Y = mvgmm.sample( N, n )
    for v in xrange( VIEWS ):
        assert( sc.allclose( X[v][:n], Y[v] ) )

def main( fname, dataset_type, k, d, params ):
    """Generate dataset in file fname"""
    if dataset_type == "gmm":
        if params.cov == "spherical" and params.sigma2 > 0:
            params.cov = array( [params.sigma2 * eye(d)] * k )
        gmm = GaussianMixtureModel.generate( fname, k, d, params.means,
                params.cov, params.weights, dirichlet_scale=params.w0, gaussian_precision = params.sigma2 )
        gmm.set_seed( params.seed )
        gmm.save() 
    elif dataset_type == "mvgmm":
        views = params.views 
        if params.cov == "spherical" and params.sigma2 > 0:
            params.cov = array( [[params.sigma2 * eye(d)] * k] * views )
        mvgmm = MultiViewGaussianMixtureModel.generate( fname, k, d, views, params.means,
                params.cov, params.weights, dirichlet_scale=params.w0, gaussian_precision = params.sigma2 )
        mvgmm.set_seed( params.seed )
        mvgmm.save()
    else:
        raise NotImplementedError

if __name__ == "__main__":
    import argparse
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument( "fname", help="Output file (as npz)" )
    parser.add_argument( "model", help="Model: gmm|mvgmm" )
    parser.add_argument( "k", type=int, help="Number of mixture components"  )
    parser.add_argument( "d", type=int, help="Dimensionality of each component"  )

    parser.add_argument( "--seed", default=int(time.time() * 1000), type=int )
    parser.add_argument( "--weights", default="uniform", help="Mixture weights, default=uniform|random" )
    parser.add_argument( "--w0", default=10.0, type=float, help="Scale parameter for the Dirichlet" )
    # GMM options
    parser.add_argument( "--means", default="hypercube", help="Mean generation procedure, default = hypercube" )
    parser.add_argument( "--cov", default="spherical", help="Covariance generation procedure, default = spherical|random"  )
    parser.add_argument( "--sigma2", default=0.2, type=float )
    # Multiview options
    parser.add_argument( "--views", default=3, help="Number of views", type=int )

    args = parser.parse_args()
    sc.random.seed( int( args.seed ) )

    main( args.fname, args.model, args.k, args.d, args )



