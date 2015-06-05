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

class GaussianMixtureModel(Model):
    """Generic mixture model with N components"""
    def __init__(self, **params):
        """Create a mixture model for components using given weights"""
        Model.__init__(self, **params)
        self.k = self["k"]
        self.d = self["d"]
        self.weights = self["w"]
        self.means = self["M"]
        self.sigmas = self["S"]
        
    @staticmethod
    def from_file(fname):
        """Load model from a HDF file"""
        model = Model.from_file(fname) 
        return GaussianMixtureModel(**model.params)

    def sample(self, N, n = -1):
        """Sample N samples from the model. If N, n are both specified,
        then generate N samples, but only keep n of them"""
        if n <= 0: 
            n = N
        shape = (n, self.d)

        #X = self._allocate_samples("X", shape)
        X = zeros(shape)
        # Get a random permutation of N elements
        perm = permutation(N)

        # Sample the number of samples from each view
        cnts = multinomial(N, self.weights)

        cnt_ = 0
        for i in xrange(self.k):
            cnt = cnts[i]
            # Generate a bunch of points for each mean
            mean, sigma = self.means.T[ i ], self.sigmas[ i ]
            perm_ = perm[cnt_ : cnt_ + cnt]
            X[perm_] = multivariate_normal(mean, sigma, cnt)
            cnt_ += cnt
        #X.flush()
        return X

    def get_exact_moments(self):
        d = self.d
        sigma2 = self.sigmas[0,0,0]
        M1 = sum(w * mu for w, mu in zip(self.weights, self.means.T)).reshape(d,1)
        M2 = sum(w * (sc.outer(mu,mu) + S) for w, mu, S in zip(self.weights, self.means.T, self.sigmas))
        M3 = sum(w * tensorify(mu,mu,mu) for w, mu, S in zip(self.weights, self.means.T, self.sigmas))

        M1_ = np.hstack((M1 for _ in range(d)))
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
            if len(covs) == 1:
                expr = expr * (GaussianMixtureModel.polymom_univariate(xis[dim-1], covs[0], deg))
            else:
                expr = expr * (GaussianMixtureModel.polymom_univariate(xis[dim-1], covs[dim-1], deg))
        return expr.expand()

    def polymom_init_symbols(self, spherical = False):
        # symbolica means and covs
        self.spherical = spherical
        self.sym_means = sp.symbols('xi1:'+str(self.d+1))
        if not spherical:
            self.sym_covs =  sp.symbols('c1:'+str(self.d+1))
        else:
            self.sym_covs =  tuple([sp.symbols('c1')])
            
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
        #ipdb.set_trace()
        allvars = self.sym_means+self.sym_covs
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
    def generate(k, d, means = "hypercube", cov = "spherical",
            weights = "random", dirichlet_scale = 10, gaussian_precision
            = 0.01):
        """Generate a mixture of k d-dimensional gaussians""" 

        params = {}

        params["k"] = k
        params["d"] = d

        if weights == "random":
            w = dirichlet(ones(k) * dirichlet_scale) 
        elif weights == "uniform":
            w = ones(k)/k
        elif isinstance(weights, sc.ndarray):
            w = weights
        else:
            raise NotImplementedError

        if means == "hypercube":
            # Place means at the vertices of the hypercube
            M = zeros((d, k))
            if k <= 2**d:
                # the minimum number of ones needed to fill k of them
                numones = int(sc.ceil(sc.log(k)/sc.log(d)))
                allinds = combinations(range(d), numones)
                for i,inds in enumerate(allinds):
                    if i == k: break
                    M[inds, i] = 1
                M = M + 0.02*sc.rand(d,k)
            else:
                raise NotImplementedError
        elif means == "hypercubenoise":
            # Place means at the vertices of the hypercube
            M = zeros((d, k))
            if k <= 2**d:
                # the minimum number of ones needed to fill k of them
                numones = int(sc.ceil(sc.log(k)/sc.log(d)))
                allinds = combinations(range(d), numones)
                for i,inds in enumerate(allinds):
                    if i == k: break
                    M[inds, i] = 1
                M = M + 0.1*sc.rand(d,k)
            else:
                raise NotImplementedError
        elif means == "constrained":
            # Place means at the vertices of the hypercube
            M = zeros((d, k))
            if k <= 2**d:
                # the minimum number of ones needed to fill k of them
                numones = int(sc.ceil(sc.log(k)/sc.log(d)))
                allinds = combinations(range(d), numones)
                for i,inds in enumerate(allinds):
                    if i == k: break
                    M[inds, i] = 1
                cnstr = 0.1*sc.rand(d,k)
                cnstrnormal = np.sum(cnstr,1)
                pert0 = cnstr - cnstrnormal
                assert(np.sum(pert0) <= 1e-5)
                M = M + pert0
            else:
                raise NotImplementedError
        elif means == "rotatedhypercube":
            # Place means at the vertices of the hypercube
            M = zeros((d, k))
            randmat = sc.randn(d,d)
            randrot,__,__ = sc.linalg.svd(randmat)
            if k <= 2**d:
                # the minimum number of ones needed to fill k of them
                numones = int(sc.ceil(sc.log(k)/sc.log(d)))
                allinds = combinations(range(d), numones)
                for i,inds in enumerate(allinds):
                    if i == k: break
                    M[inds, i] = 1
            else:
                raise NotImplementedError
            M = randrot.dot(M)
                
        elif means == "random":
            M = 5*sc.randn(d, k)
        elif isinstance(means, sc.ndarray):
            M = means
        else:
            raise NotImplementedError

        if cov == "spherical":
            # Using 1/gamma instead of inv_gamma
            sigmas = []
            for i in xrange(k):
                sigmak = 2*sc.random.rand()+2
                sigmas = sigmas + [ sigmak * eye(d) ]
            S = array(sigmas)

        elif cov == "spherical_uniform":
            # Using 1/gamma instead of inv_gamma
            sigma = 1/sc.random.gamma(1/gaussian_precision)
            S = array([ sigma * eye(d) for i in xrange(k) ])
        elif cov == "diagonal":
            # Using 1/gamma instead of inv_gamma
            sigmas = []
            for i in xrange(k):
                sigmak = [3*sc.random.rand()+1 for i in xrange(d)]
                sigmas = sigmas + [ sc.diag(sigmak) ]
            S = array(sigmas)
        elif isinstance(cov, sc.ndarray):
            S = cov
        elif cov == "random":
            S = array([ gaussian_precision * inv(wishart(d+1, sc.eye(d), 1)) for i in xrange(k) ])
        else:
            raise NotImplementedError

        params["w"] = w
        params["M"] = M
        params["S"] = S

        # Unwrap the store and put it into the appropriate model
        return GaussianMixtureModel(**params)

    def get_log_likelihood(self, X):
        lhood = 0.
        for x in X:
            lhood_ = 0.
            for i in xrange(self.k):
                lhood_ += self.weights[i] * sc.exp(-0.5 * (self.means[i] - x).dot(self.sigmas[i]).dot(self.means[i] - x))
            lhood += sc.log(lhood_)
        return lhood

def test_gaussian_mixture_generator_dimensions():
    "Test the GaussianMixtureModel generator"
    import tempfile
    fname = tempfile.mktemp()

    N = 1000
    D = 10
    K = 3

    gmm = GaussianMixtureModel.generate(fname, K, D)
    assert(gmm.means.shape == (D, K))
    assert(gmm.weights.shape == (K,))

    X = gmm.sample(N)
    assert(X.shape == (N, D))

def test_gaussian_mixture_generator_replicatability():
    "Test the GaussianMixtureModel generator"
    import tempfile
    fname = tempfile.mktemp()

    N = 1000
    n = 500
    D = 10
    K = 3

    gmm = GaussianMixtureModel.generate(fname, K, D)
    gmm.set_seed(100)
    gmm.save()

    X = gmm.sample(N)
    del gmm

    gmm = GaussianMixtureModel.from_file(fname)
    Y = gmm.sample(N)
    assert(sc.allclose(X, Y))
    del gmm

    gmm = GaussianMixtureModel.from_file(fname)
    Y = gmm.sample(N, n)
    assert(sc.allclose(X[:n], Y))
