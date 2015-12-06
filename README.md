# polymom
#### Estimating Mixture Models via Mixtures of Polynomials
Sida Wang, Percy Liang, Arun Chaganty

Provide a list of polynomial moment constraints, and recovery parameters of mixture models. See our NIPS paper for details.

The ipython notebooks [MixtureOfGaussians.ipynb](https://github.com/sidaw/polymom/blob/master/MixtureOfGaussians.ipynb)
and [MixtureLinearRegressions.ipynb](https://github.com/sidaw/polymom/blob/master/MixtureLinearRegressions.ipynb)
shows how to use polymom for these mixture models.

requires [cvxopt](http://cvxopt.org/).



### About

**mompy**: is a package for building the moment matrix, solving the sdp (requires cvxopt), and extracting solutions
```python
# construct the degree d moment matrix with the provided symbols
MM = mp.MomentMatrix(d, symbols, morder='grevlex') 
# generate an SDP and calls cvxopt
solsdp = mp.solvers.solve_basic_constraints(MM, constraints, 1e-8); 
sol = mp.extractors.extract_solutions_lasserre(MM, solsdp['x'], Kmax = k)
```

**models**: contains code to generate Gaussian, multiview, MLR models, sampling from them, and obtaining the list of variables.

**archive**: experimental stuff

**algos**: contains spectral methods for estimating mixture models
