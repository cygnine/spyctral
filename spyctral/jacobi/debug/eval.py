#!/usr/bin/env python
"""
* File Name : eval.py

* Creation Date : 2009-06-17

* Created By : Akil Narayan

* Last Modified : Wed 17 Jun 2009 01:54:48 PM EDT

* Purpose : Debugging for evaluation of monic, normalized Jacobi polynomials
"""

import numpy as np
from numpy.linalg import norm
import scipy as sp
from scipy import rand
import pylab as pl
import opoly1.jacobi as jac_ref
import pyspec.jacobi as jac

N = 10
Nx = 100
alpha = -1 + 11*rand();
beta = -1 + 11*rand();
x = np.linspace(-1,1,Nx)
n = range(N)

# Testing out monic, normalized polynomial evaluation
poly_ref = jac_ref.jpoly(x,n,alpha=alpha,beta=beta)
polyn_ref = jac_ref.jpolyn(x,n,alpha=alpha,beta=beta)
[aref,bref] = jac_ref.recurrence(N,alpha=alpha,beta=beta)

poly = jac.eval.jpoly(x,n,alpha=alpha,beta=beta,normalization='monic')
polyn = jac.eval.jpoly(x,n,alpha=alpha,beta=beta,normalization='normal')
[a,b] = jac.coeffs.recurrence_range(N,alpha=alpha,beta=beta)

print "Monic error is ", norm(poly-poly_ref)
print "Normalized error is ", norm(polyn-polyn_ref)
