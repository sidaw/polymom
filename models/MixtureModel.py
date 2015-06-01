"""
Generate data from a three-view mixture model
"""

import numpy as np
import scipy as sc
import scipy.linalg
from scipy import array, zeros, ones, eye, allclose
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

class MixtureModel(Model):
    """Generic mixture model with 3 components"""
    def __init__(self, **params):
        """Create a mixture model for components using given weights"""
        Model.__init__(self, **params)
        self.k = self["k"]
        self.d = self["d"]
        self.weights = self["w"]
        self.means = self["M"] # Draw as a multinomial distribution

        assert allclose(self.weights.sum(), 1.)
        assert allclose(self.means.sum(0), 1.)

        # symbolic means and observed variables
        self.sym_means = sp.symbols('x1:'+str(self.d+1))
        self.sym_obs = self.sym_means

    @staticmethod
    def from_file(fname):
        """Load model from a HDF file"""
        model = Model.from_file(fname)
        return MixtureModel(**model.params)

    def sample(self, N):
        """Sample N samples from the model. If N, n are both specified,
        then generate N samples, but only keep n of them"""
        shape = (N, self.d, 3)

        X = zeros(shape)
        #X = self._allocate_samples("X", shape)
        # Get a random permutation of N elements
        perm = permutation(N)

        # Sample the number of samples from each view
        cnts = multinomial(N, self.weights)

        cnt_ = 0
        for i in xrange(self.k):
            cnt = cnts[i]
            # Generate a bunch of points for each mean
            mean = self.means.T[i]
            perm_ = perm[cnt_ : cnt_ + cnt]
            X[perm_,:,0] = multinomial(1, mean, size=int(cnt))
            X[perm_,:,1] = multinomial(1, mean, size=int(cnt))
            X[perm_,:,2] = multinomial(1, mean, size=int(cnt))

            cnt_ += cnt
        return X

    def llikelihood(self, xs):
        lhood = 0.
        for i, x in enumerate(xs):
            x1, x2, x3 = x.argmax(0)

            lhood_ = 0.
            for i in xrange(self.k):
                lhood_ += self.weights[i] * self.means[x1,i] * self.means[x2,i] * self.means[x3,i]
            lhood += (sc.log(lhood_) - lhood)/(i+1)
        return lhood

    def _compute_power(self, term, x):
        """Compute a term to a power"""
        powers = term.as_powers_dict()
        moment = 1.
        for i, s in enumerate(self.sym_means):
            moment *= x[i]**(powers[s])
        return moment

    def _compute_empirical_power(self, term, x):
        """Assumes enough views to compute powers"""
        powers = term.as_powers_dict()
        moment = 1.
        idx = 0
        for i, s in enumerate(self.sym_means):
            for _ in xrange(powers[s]):
                moment *= x[i,idx]
                idx += 1
        return moment

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

    def _moment_equations(self, term):
        """
        Get expression for term - in this case, it's really easy, it's just the term itself.
        """
        return term

    def exact_moment_equations(self, maxdeg):
        """
        return scipy moment equation expressions
        """
        terms = self.observed_monomials(maxdeg)
        moments = self.exact_moments(terms)

        return [self._moment_equations(term) - moments[term] for term in terms]

    def empirical_moment_equations(self, xs, maxdeg):
        """
        return scipy moment equation expressions
        """
        terms = self.observed_monomials(maxdeg)
        moments = self.empirical_moments(xs, terms)

        return [self._moment_equations(term) - moments[term] for term in terms]

    def moment_monomials(self, maxdeg):
        """
        return scipy moment monomials
        """
        import sympy.polys.monomials as mn
        allvars = self.sym_obs
        terms = mn.itermonomials(allvars, maxdeg)
        return terms

    def observed_monomials(self, maxdeg):
        """
        return scipy moment monomials
        """
        import sympy.polys.monomials as mn
        obsvars = self.sym_obs
        terms = mn.itermonomials(obsvars, maxdeg)
        return terms

    @staticmethod
    def generate(k, d, means = "random", weights = "random", dirichlet_scale = 1.):
        """Generate a mixture of k d-dimensional gaussians"""

        params = {}

        params["k"] =  k
        params["d"] =  d

        if weights == "random":
            w = dirichlet(ones(k) * dirichlet_scale)
        elif weights == "uniform":
            w = ones(k)/k
        elif isinstance(weights, sc.ndarray):
            w = weights
        else:
            raise NotImplementedError

        if means == "random":
            M = dirichlet(ones(d) * dirichlet_scale, k).T
        elif isinstance(means, sc.ndarray):
            M = means
        else:
            raise NotImplementedError

        params["w"] =  w
        params["M"] =  M

        # Unwrap the store and put it into the appropriate model
        return MixtureModel(**params)

    def using(self, **kwargs):
        """Create a new model with modified parameters"""
        params_ = dict(self.params)
        params_.update(**kwargs)
        return MixtureModel(**params_)


