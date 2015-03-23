# Percy's playground.

from __future__ import print_function
import sympy as sp
import numpy as np
import BorderBasis as BB
np.set_printoptions(precision=3)
from IPython.display import display, Markdown, Math
sp.init_printing()

R, x, y = sp.ring('x,y', sp.RR, order=sp.grevlex)
I = [ x**2 + y**2 - 1.0, x + y ]

R, x, y, z = sp.ring('x,y,z', sp.RR, order=sp.grevlex)
I = [ x**2 - 1, y**2 - 4, z**2 - 9]

# n = 4 takes a long time
n = 2
Rvs = sp.ring(' '.join('v'+str(i) for i in range(1, n + 1)), sp.RR, order=sp.grevlex)
R, vs = Rvs[0], Rvs[1:]
I = [v**2 - 1 for v in vs]

B = BB.BorderBasisFactory(1e-5).generate(R,I)

print("=== Generator Basis:")
for f in B.generator_basis:
    display(f.as_expr())

print("=== Quotient Basis:")
for f in B.quotient_basis():
    display(f.as_expr())

# v2 is always zero
print("=== Variety:")
for v in B.zeros():
    print(zip(R.symbols, v))
