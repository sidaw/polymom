"""
Generate data from a mixture of linear regressions
"""

import sympy as sp
import scipy as sc
import scipy.linalg
from scipy import array, zeros, ones, eye, allclose, ndarray, log, exp, sqrt
from scipy.linalg import inv
from models.Model import Model

multinomial = sc.random.multinomial
multivariate_normal = sc.random.multivariate_normal
dirichlet = sc.random.dirichlet
randn = sc.random.randn
gamma = sc.random.gamma

from util import permutation

class LinearRegressionsMixture(Model):
    """Generic mixture model of 3 linear regressions"""

    def __init__(self, **params):
        """Create a mixture model for components using given weights"""
        Model.__init__(self, **params)
        self.k = self["k"]
        self.d = self["d"]

        self.weights = self["w"]
        self.betas = self["B"] # Draw as a multinomial distribution

        assert allclose(self.weights.sum(), 1.)

        self.mean = self["xM"]
        self.sigma = self["xS"]
        self.sym_betas = sp.symbols('b1:'+str(self.d+1))
        self.sym_obs = sp.symbols('x1:'+str(self.d+1) + 'y')

    @staticmethod
    def from_file(fname):
        """Load model from a HDF file"""
        model = Model.from_file(fname)
        return LinearRegressionsMixture(**model.params)

    def sample(self, N):
        """Sample N samples from the model. If N, n are both specified,
        then generate N samples, but only keep n of them"""

        # Get a random permutation of N elements
        perm = permutation(N)

        # Sample the number of samples from each view
        cnts = multinomial(N, self.weights)

        # Generate Xs
        xs = multivariate_normal(self.mean, self.sigma, N)
        ys = zeros((N,))

        cnt_ = 0
        for i in xrange(self.k):
            cnt = cnts[i]
            # Generate a bunch of points for each mean
            beta = self.betas.T[i]

            perm_ = perm[cnt_ : cnt_ + cnt]
            ys[perm_] = xs[perm_].dot(beta)
        return xs, ys

    def llikelihood(self, xs):
        """Compute llikelihood - not fully normalized"""
        lhood = 0.
        for i, (x, y) in enumerate(xs):
            lhood_ = 0.
            for i in xrange(self.k):
                lhood_ += self.weights[i] * 1./sqrt(2) * exp(-0.5 * (self.betas.T[i].dot(x) - y)**2)
            lhood += log(lhood_)
        return lhood

    def exact_moments(self, terms):
        """
        Get the exact moments corresponding to a particular term
        """

        terms = [sp.sympify(term) if isinstance(term, str) else term for term in terms]

        moment = {term : 0. for term in terms}
        for k in xrange(self.k):
            for term in terms:
                moment[term] += self.weights[k] * self._compute_power(term, self.means.T[k])
        return moment

    def empirical_moments(self, xs, terms):
        """
        Get the exact moments corresponding to a particular term
        """
        moment = {term : 0. for term in terms}
        for i, x in enumerate(xs):
            for term in terms:
                moment[term] += (self._compute_empirical_power(term, x) - moment[term])/(i+1)
        return moment



    @staticmethod
    def generate(k, d, mean = "zero", cov = "random", betas = "random", weights = "random",
            dirichlet_scale = 1., gaussian_precision = 0.1):
        """Generate a mixture of k d-dimensional multi-view gaussians"""

        params = {}
        params["k"] =  k
        params["d"] =  d

        if weights == "random":
            w = dirichlet(ones(k) * dirichlet_scale)
        elif weights == "uniform":
            w = ones(k)/k
        elif isinstance(weights, ndarray):
            w = weights
        else:
            raise NotImplementedError

        if betas == "eye":
            B = eye(d)[:,:k]
        elif betas == "random":
            B = randn(d, k)
        elif isinstance(betas, ndarray):
            B = betas
        else:
            raise NotImplementedError

        if mean == "zero":
            M = zeros(d)
        elif mean == "random":
            M = randn(d)
        elif isinstance(mean, ndarray):
            M = mean
        else:
            raise NotImplementedError

        if cov == "eye":
            S = eye(d)
        elif cov == "spherical":
            # Using 1/gamma instead of inv_gamma
            sigma = 1/gamma(1/gaussian_precision)
            S = sigma * eye(d)
        #elif cov == "random":
        #    S = gaussian_precision * inv(wishart(d+1, sc.eye(d), 1))
        elif isinstance(cov, ndarray):
            S = cov
        else:
            raise NotImplementedError

        params["w"] = w
        params["B"] = B
        params["M"] = M
        params["S"] = S

        # Unwrap the store and put it into the appropriate model
        return LinearRegressionsMixture(**params)

