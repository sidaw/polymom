
# coding: utf-8

# # Mixture of Gaussians
# 
# In this note, we will solve the mixture of Gaussians

# This block contains some $\LaTeX$ macros.
# $\newcommand{\E}{\mathbb{E}}$
# $\renewcommand{\Re}{\mathbb{R}}$
# $\newcommand{\oft}[1]{{^{(#1)}}}$
# $\newcommand{\oftt}[2]{{^{(#1)#2}}}$

# In[6]:

import util
from util import *
import numpy as np
import models
import mompy as mp
import cvxopt

np.set_printoptions(precision=3, suppress=True)
from IPython.display import display, Markdown, Math
from operator import mul
import sympy as sp


sp.init_printing()


# ## Noiseless Example
# 
# Let's construct a toy example for the rest of this document.
# 
# Let $\xi_1, ..., \xi_K \in \Re^D$ be the set of means. Let $c_1, ..., c_K \in \Re^D$ be the set of diagonal covariances.

# In[25]:

k = 3
d = 3
degobs = 4
degmm = 3

#sc.random.seed(11)
gm = models.GaussianMixtureModel.generate('', k, d, means='rotatedhypercube', cov='diagonal')

monos = gm.polymom_monos(degmm)
constraints = gm.polymom_all_constraints(degobs)
xis = gm.sym_means
covs = gm.sym_covs
sym_all = xis + covs
print 'The polynomial constraints are as follows'
display(constraints)


# With this machinery, we can compute the moment polynomials required for the moment method magic!

# In[26]:

MM = mp.MomentMatrix(degmm, sym_all, morder='grevlex', monos=monos)
display(MM.row_monos)


# In[38]:

# cin = mp.solvers.get_cvxopt_inputs(MM, constraints)
# Bf = MM.get_Bflat()
# R = np.random.rand(len(MM), len(MM))
# W = R.dot(R.T)
# W = np.eye(len(MM))
# for i in xrange(1):
#    w = Bf.dot(W.flatten())
#    solsdp = cvxopt.solvers.sdp(cvxopt.matrix(w), Gs=cin['G'], hs=cin['h'], A=cin['A'], b=cin['b']) 
# solsdp = cvxopt.solvers.sdp(cin['c'], Gs=cin['G'], hs=cin['h'], A=cin['A'], b=cin['b'])
solsdp = mp.solvers.solve_moments_with_convexiterations(MM, constraints, k, maxiter = 50);



# In[24]:

sol_lasserre = mp.extractors.extract_solutions_lasserre(MM, solsdp['x'], Kmax = k)
sol_dreesen = mp.extractors.extract_solutions_dreesen(MM, solsdp['x'], Kmax = k)
trueparams = {}

for i in xrange(d):
    trueparams[xis[i]] = []
    trueparams[covs[i]] = []
    for j in xrange(k):
        trueparams[xis[i]].append(gm.means[i,j])
        trueparams[covs[i]].append(gm.sigmas[j,i,i])
print 'the true parameters'
display(trueparams)
print 'recovered parameters (Lasserre)'
display(sol_lasserre)
print 'recovered parameters (Dreesen)'
display(sol_dreesen)


# ## Now take samples!
# 

# In[22]:

num_samples = 1000
X = gm.sample(num_samples)
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

import numpy.linalg

# plot after a random orthogonal projection
randdir,__,__ = np.linalg.svd(np.random.randn(d,2))
Y = X.dot(randdir[:,0:2])
plt.scatter(X[:,0], X[:,1]);

constraints_noisy = gm.polymom_all_constraints_samples(degobs, X)
display(constraints_noisy)


# In[24]:

cin = mp.solvers.get_cvxopt_inputs(MM, constraints_noisy)
solsdp_noisy = cvxopt.solvers.sdp(cin['c'], Gs=cin['G'], hs=cin['h'], A=cin['A'], b=cin['b'])
print solsdp_noisy


# In[21]:

sol_lasserre_noisy = mp.extractors.extract_solutions_lasserre(MM, solsdp_noisy['x'], Kmax = k)
sol_dreesen_noisy = mp.extractors.extract_solutions_dreesen_proto(MM, solsdp_noisy['x'], Kmax = k)

print 'the true parameters'
display(trueparams)
print 'recovered parameters (Lasserre)'
display(sol_lasserre_noisy)
print 'recovered parameters (Dreesen)'
display(sol_dreesen_noisy)


# ## EM Algorithm

# In[22]:

from algos import GaussianMixturesEM
reload(GaussianMixturesEM)
algo = GaussianMixturesEM.GaussianMixtureEM( k, d )
X.shape
lhood, Z, O_ = algo.run( X, None )


# In[23]:

O_, gm.weights


# In[24]:

gm.sigmas


# In[25]:

k, d, M, S, w = gm.k, gm.d, gm.means, gm.sigmas, gm.weights
import algos.GaussianMixturesSpectral as gms
M_ = gms.find_means(X, k)
print "M_", M_.T


# In[ ]:




# In[ ]:




# In[ ]:



