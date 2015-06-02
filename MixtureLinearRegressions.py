
# coding: utf-8

# # Mixture of Linear Regressions
# 
# In this note, we will solve the mixture of linear regressions problem using the moment method.

# This block contains some $\LaTeX$ macros.
# $\newcommand{\E}{\mathbb{E}}$
# $\renewcommand{\Re}{\mathbb{R}}$
# $\newcommand{\oft}[1]{{^{(#1)}}}$
# $\newcommand{\oftt}[2]{{^{(#1)#2}}}$

# In[1]:

import util
from util import *
from models import LinearRegressionsMixture
import numpy as np
np.set_printoptions(precision=3, suppress=True)
from IPython.display import display, Markdown, Math
from operator import mul
import sympy as sp
sp.init_printing()


# Let $w_1, ..., w_K \in \Re^D$ be a set of $K$ regression coefficients. Let $x_1, ..., x_N \in \Re^D$ be a given set of data points and let $y_n = \sum_{k} \delta_k w_k^\top x_n + \epsilon$ be observed responses. Our objective is to recover $(w_k)$.

# ## Toy Example
# 
# Let's construct a toy example for the rest of this document.

# In[2]:

K, D = 2, 2
np.random.seed(0)
pis = np.array([0.4, 0.6])
ws = np.array([[0.75, 0.25], [0.4, 0.9]])
model = LinearRegressionsMixture.generate("tmp.dat", K, D, betas = ws, weights = pis, cov = 1*np.eye(D))
print("True parameters:")
print(model.weights)
print(model.betas)


# # Noiseless, Infinite Data Setting
# 
# In this scenario, assume that $\epsilon = 0$ and that we observe expected moments of the data. Consider the moments:
# \begin{align}
# \E[x^\alpha y^b] 
# &= \sum_k \pi_k \E[x^\alpha y^b | \delta_k = 1] \\
# &= \sum_k \pi_k \E[x^\alpha [w_k^\top x]^b] \\
# &= \sum_k \pi_k \E[x^\alpha \sum_{\beta \in p(b)} w_k^\beta x^\beta] \\
# &= \sum_{\beta \in p(b)} \E_\pi[w^\beta] \E[x^{\alpha + \beta}],
# \end{align}
# where $p(b)$ are the $d$-partitions of $b$. Note that $\E[x^{\alpha + \beta}]$ are observable quantities, and thus this simply represents a mixture over the polynomials with terms $w_k^\beta$.

# In[3]:

def evaluate_mixture(ws, pis, beta):
    """
    Compute E_\pi[w^beta]
    """
    return sum(pi * util.monomial(w, beta) for w, pi in zip(ws.T, pis))
    
def compute_exact_y_moments(ws, pis, moments_x, alpha, b):
    """
    Compute the exact moments E[x^a y^b] using coefficients ws, pis and moments_x.
    """
    D, K = ws.shape
    ret = 0.
    for beta in partitions(D, b):
        ret += evaluate_mixture(ws, pis, beta) * moments_x[tuple_add(alpha, beta)]
    return ret

def describe_moment_polynomial(R, moments_x, moment_y, alpha, b):
    """
    Computes the moment polynomial for E[x^alpha, y^b]
    """
    D = len(R.symbols)
    w = R.symbols
    expr = -moment_y
    for beta in partitions(D, b):
        expr += util.monomial(w, beta) * moments_x[tuple_add(alpha, beta)]
    return expr


# In[4]:

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
    return {alpha : 1.}
def expected_moments_y(ws, pis, moments_x, alphabs):
    return {(alpha, b) : compute_exact_y_moments(ws, pis, moments_x, alpha, b) for alpha, b in alphabs}


# In[5]:

R, _ = sp.xring(['w%d'%d for d in xrange(D)], sp.RR, sp.grevlex)

deg_b, deg_x = 3, 3
sigma = 1.0
alphas = list(dominated_elements((deg_x for _ in xrange(D))))
alphabs = [(alpha, b) for alpha in alphas for b in xrange(1,deg_b+1)]
alphas_ = list(dominated_elements((deg_x + deg_b for _ in xrange(D))))
moments_x = expected_gaussian_moments(sigma, alphas_)
moments_y = expected_moments_y(ws, pis, moments_x, alphabs)
#display(moments)


# In[6]:

def get_constraint_polynomials(moments_y, moments_x, deg_x, deg_b):
    constrs = []
    for b in xrange(1, deg_b+1):
        for alpha in util.dominated_elements((deg_x for _ in xrange(D))):
            constrs.append( describe_moment_polynomial(R, moments_x, moments_y[(alpha, b)], alpha, b) )
    return constrs


# With this machinery, we can compute the moment polynomials required for the moment method magic!

# In[7]:

from mompy.core import MomentMatrix
import mompy.solvers as solvers; reload(solvers)
import mompy.extractors as extractors; reload(extractors)

constrs_exact = get_constraint_polynomials(moments_y, moments_x, deg_x, deg_b)
M = MomentMatrix(3, R.symbols, morder='grevlex')
sol = solvers.solve_generalized_mom_coneqp(M, constrs_exact, None)
#sol = solvers.solve_basic_constraints(M, constrs_exact, slack=0)



# In[8]:

display( model.betas)
display(extractors.extract_solutions_lasserre(M, sol['x'], Kmax=2))
display(extractors.extract_solutions_dreesen_proto(M, sol['x'], Kmax=2))


# ## With real samples
# 
# Let's try to solve the problem with generated samples.
# 

# In[9]:

def compute_expected_moments(xs, alphas):
    moments = {alpha : 0. for alpha in alphas}
    for i, x in enumerate(xs):
        for alpha in alphas:
            moments[alpha] += (monomial(x, alpha) - moments[alpha])/(i+1)
    return moments

def compute_expected_moments_y(ys, xs, alphabs):
    moments = {(alpha, b) : 0. for alpha, b in alphabs}
    for i, (y, x) in enumerate(zip(ys,xs)):
        for alpha, b in alphabs:
            moments[(alpha,b)] += (monomial(x, alpha) * y**b - moments[(alpha,b)])/(i+1)
    return moments


# In[10]:

ys, xs = model.sample(1e3)
moments_x = compute_expected_moments(xs, alphas_)
moments_y = compute_expected_moments_y(ys, xs, alphabs)


# In[11]:


from mompy.core import MomentMatrix
import mompy.solvers as solvers; reload(solvers)
import mompy.extractors as extractors; reload(extractors)

constrs_noisy = get_constraint_polynomials(moments_y, moments_x, deg_x, deg_b)
#display(constrs_noisy)
M = MomentMatrix(2, R.symbols, morder='grevlex')
solsdp = solvers.solve_generalized_mom_coneqp(M, constrs_noisy, None)
#solsdp = solvers.solve_basic_constraints(M, constrs_noisy, slack=2e-2)
#sol = solvers.solve_moments_with_convexiterations(M, constrs, 3)
print solsdp['x']


# In[12]:

display(model.betas)
display(extractors.extract_solutions_lasserre(M, solsdp['x'], Kmax=2))

display(extractors.extract_solutions_dreesen_proto(M, solsdp['x'], Kmax=2))
sol = extractors.extract_solutions_dreesen_proto(M, solsdp['x'], Kmax=2)
ws_rec = array([sol[sp.symbols('w0')], sol[sp.symbols('w1')]])


# In[13]:

for i in xrange(10):
    print constrs_exact[i];
    print constrs_noisy[i];
    print '\n'


# In[14]:

M.numeric_instance(solsdp['x'])


# In[15]:

import scipy as sc
import numpy.linalg
np.linalg.matrix_rank((M.numeric_instance(solsdp['x'])), 1e-2)


# In[16]:

M.row_monos


# In[17]:

M.numeric_instance(solsdp['x'])


# In[18]:

moments_y


# In[19]:

moments_x_true = expected_gaussian_moments(sigma, alphas_)
moments_y_true = expected_moments_y(ws, pis, moments_x, alphabs)


# In[20]:

for key in moments_y:
    display(key, abs(moments_y[key] - moments_y_true[key]))

