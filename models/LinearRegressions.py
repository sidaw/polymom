"""
Generate data from a mixture of linear regressions
"""
import ipdb

import sympy as sp
from sympy import sympify
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

from util import permutation, monomial, partitions, tuple_add, prod, dominated_elements

from operator import mul

def evaluate_mixture(ws, pis, beta):
    r"""
    Compute E_\pi[w^beta]
    """
    return sum(pi * monomial(w, beta) for w, pi in zip(ws.T, pis))

def compute_exact_y_moments(ws, pis, moments_x, alpha, b):
    """
    Compute the exact moments E[x^a y^b] using coefficients ws, pis and moments_x.
    """
    D, _ = ws.shape
    coeffs = sp.ntheory.multinomial_coefficients(D, b)

    ret = 0.
    for beta in partitions(D, b):
        ret += coeffs[beta] * evaluate_mixture(ws, pis, beta) * moments_x[tuple_add(alpha, beta)]
    return ret

def describe_moment_polynomial(syms, moments_x, moment_y, alpha, b):
    """
    Computes the moment polynomial for E[x^alpha, y^b]
    """
    D = len(syms)
    expr = -moment_y
    coeffs = sp.ntheory.multinomial_coefficients(D, b)
    for beta in partitions(D, b):
        expr += coeffs[beta] * monomial(syms, beta) * moments_x[tuple_add(alpha, beta)]
    return expr

# Example 
def double_factorial(n): 
    return reduce(mul, xrange(n, 0, -2)) if n > 0 else 1
def gaussian_moments(sigma, d):
    """
    E[x^d] where x is standard gaussian with sigma
    """
    if d == 0: return 1
    elif d % 2 == 0: return double_factorial(d-1) * sigma**d
    else: return 0
def expected_gaussian_moments(sigma, alphas):
    return {alpha : prod(gaussian_moments(sigma, a) for a in alpha) for alpha in alphas}
def expected_uniform_moments(alphas):
    return {alpha : 1. for alpha in alphas}
def exact_moments_y(ws, pis, moments_x, alphabs):
    return {(alpha, b) : compute_exact_y_moments(ws, pis, moments_x, alpha, b) for alpha, b in alphabs}

def compute_expected_moments(xs, alphas):
    moments = {alpha : 0. for alpha in alphas}
    for alpha in alphas:
        m = monomial(xs.T, alpha)
        moments[alpha] = m if isinstance(m,float) else float(m.mean())
    return moments

def compute_expected_moments_y(ys, xs, alphabs):
    moments = {(alpha, b) : 0. for alpha, b in alphabs}
    for alpha, b in alphabs:
        moments[(alpha,b)] = float((monomial(xs.T, alpha) * ys**b).mean())
    return moments

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
        self.sigma_val = self["xSigma"]
        self.sym_betas = sp.symbols('b1:'+str(self.d+1))
        self.sym_obs = sp.symbols('x1:'+str(self.d+1) + 'y')

    def param_symbols(self):
        """Symbols"""
        return self.sym_betas

    @staticmethod
    def from_file(fname):
        """Load model from a HDF file"""
        model = Model.from_file(fname)
        return LinearRegressionsMixture(**model.params)

    def sample(self, N):
        """Sample N samples from the model. If N, n are both specified,
        then generate N samples, but only keep n of them"""
        N = int(N)

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
            cnt_ += cnt
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

    def _y_power(self, term):
        """Get powers of y from term"""
        powers = term.as_powers_dict()
        return powers[sympify('y')]

    def _x_power(self, term):
        """Get powers of x from term"""

        powers = term.as_powers_dict()
        return tuple(powers[sympify('x%d'%i)] for i in xrange(self.d))

    def exact_moments(self, terms):
        """
        Get the exact moments corresponding to a particular term
        """
        terms = [sympify(term) if isinstance(term, str) else term for term in terms]

        # Get max degree of b and x in terms
        D = self.d
        ws, pis = self.betas, self.weights
        deg_b = max(self._y_power(term) for term in terms)
        deg_x = max(max(self._x_power(term) for term in terms))

        sigma = self.sigma_val
        alphas = list(dominated_elements((deg_x for _ in xrange(D))))
        alphabs = [(alpha, b) for alpha in alphas for b in xrange(1,deg_b+1)]
        alphas_ = list(dominated_elements((deg_x + deg_b for _ in xrange(D))))
        moments_x = expected_gaussian_moments(sigma, alphas_)
        moments_y = exact_moments_y(ws, pis, moments_x, alphabs)

        moment = {term : 0. for term in terms}
        for term in terms:
            b = self._y_power(term)
            alpha = self._x_power(term)
            moment[term] = moments_y[(alpha,b)]
        return moment

    def empirical_moments(self, xs, terms):
        """
        Get the exact moments corresponding to a particular term
        """
        terms = [sympify(term) if isinstance(term, str) else term for term in terms]

        xs, ys = xs

        D = self.d
        deg_b = max(self._y_power(term) for term in terms)
        deg_x = max(max(self._x_power(term) for term in terms))

        alphas = list(dominated_elements((deg_x + deg_b for _ in xrange(D))))
        alphas = list(dominated_elements((deg_x for _ in xrange(D))))
        alphabs = [(alpha, b) for alpha in alphas for b in xrange(1,deg_b+1)]
        moments_y = compute_expected_moments_y(ys, xs, alphabs)

        moment = {term : 0. for term in terms}
        for term in terms:
            alpha, b = self._x_power(term), self._y_power(term)
            moment[term] = moments_y[(alpha,b)]
        return moment

    def exact_moment_equations(self, maxdeg):
        """
        return scipy moment equation expressions
        """
        D = self.d
        ws, pis = self.betas, self.weights
        deg_x, deg_b = maxdeg

        sigma = self.sigma_val
        alphas = list(dominated_elements((deg_x + deg_b for _ in xrange(D))))
        alphas = list(dominated_elements((deg_x for _ in xrange(D))))
        alphabs = [(alpha, b) for alpha in alphas for b in xrange(1,deg_b+1)]
        alphas_ = list(dominated_elements((deg_x + deg_b for _ in xrange(D))))
        moments_x = expected_gaussian_moments(sigma, alphas_)
        moments_y = exact_moments_y(ws, pis, moments_x, alphabs)

        constrs = []
        con = {}
        for b in xrange(1, deg_b+1):
            for alpha in dominated_elements((deg_x for _ in xrange(D))):
                eqn = describe_moment_polynomial(self.sym_betas, moments_x, moments_y[(alpha, b)], alpha, b)
                con[(alpha,b)] = eqn
                constrs.append(eqn)
        return constrs, con

    def empirical_moment_equations(self, xs, maxdeg):
        """
        return scipy moment equation expressions
        """
        xs, ys = xs

        D = self.d
        ws, pis = self.betas, self.weights
        deg_x, deg_b = maxdeg

        alphas = list(dominated_elements((deg_x + deg_b for _ in xrange(D))))
        alphas = list(dominated_elements((deg_x for _ in xrange(D))))
        alphabs = [(alpha, b) for alpha in alphas for b in xrange(1,deg_b+1)]
        alphas_ = list(dominated_elements((deg_x + deg_b for _ in xrange(D))))
        moments_x = compute_expected_moments(xs, alphas_)
        moments_y = compute_expected_moments_y(ys, xs, alphabs)

        constrs = []
        for b in xrange(1, deg_b+1):
            for alpha in dominated_elements((deg_x for _ in xrange(D))):
                constrs.append(describe_moment_polynomial(self.sym_betas, moments_x, moments_y[(alpha, b)], alpha, b))
        return constrs

    def moment_monomials(self, maxdeg):
        """
        return scipy moment monomials
        """
        import sympy.polys.monomials as mn
        allvars = self.sym_betas
        terms = mn.itermonomials(allvars, maxdeg)
        return terms

    def observed_monomials(self, maxdeg):
        """
        return scipy moment monomials
        """
        D = self.d
        deg_x, deg_b = maxdeg
        alphas = list(dominated_elements((deg_x for _ in xrange(D))))
        alphabs = [(alpha, b) for alpha in alphas for b in xrange(1,deg_b+1)]
        x = [sympify("x%d"%i) for i in xrange(D)]
        y = sympify("y")
        terms = map(sympify, [monomial(x, alpha) * y**b for (alpha,b) in alphabs])

        return terms

    @staticmethod
    def generate(k, d, mean = "zero", cov = 0.1, betas = "random", weights = "random",
            dirichlet_scale = 1.):
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

        if isinstance(cov, float):
            S = cov * eye(d)
        #if cov == "eye":
        #    S = eye(d)
        #elif cov == "spherical":
        #    # Using 1/gamma instead of inv_gamma
        #    sigma = 1/gamma(1/gaussian_precision)
        #    S = sigma * eye(d)
        ##elif cov == "random":
        ##    S = gaussian_precision * inv(wishart(d+1, sc.eye(d), 1))
        #elif isinstance(cov, ndarray):
        #    S = cov
        else:
            raise NotImplementedError

        params["w"] = w
        params["B"] = B
        params["xM"] = M
        params["xS"] = S
        params["xSigma"] = cov

        # Unwrap the store and put it into the appropriate model
        return LinearRegressionsMixture(**params)

