#!/usr/bin/env python
"""
* File Name : quad.py

* Creation Date : 2009-06-17

* Created By : Akil Narayan

* Last Modified : Wed 17 Jun 2009 04:29:25 PM EDT

* Purpose : Quadrature debugging
"""

import numpy as np
from numpy import sum
from numpy.linalg import norm
import scipy as sp
from scipy import rand
import pyspec as ps
import opoly1.jacobi as jac

N = 100
alpha = -1/2. + 11*rand();
beta = -1/2. + 11*rand();

#### First compare against established opoly1 module
[x_ref,w_ref] = jac.gquad(N,a=alpha,b=beta)
[x,w] = ps.jacobi.quad.gq(N,alpha=alpha,beta=beta)

print "Gauss quadrature comparison error is ", norm(x-x_ref) + norm(w-w_ref)

[x_ref,w_ref] = jac.grquad(N,a=alpha,b=beta)
[x,w] = ps.jacobi.quad.grq(N,alpha=alpha,beta=beta)

print "Gauss-Radau quadrature comparison error is ", norm(x-x_ref) + norm(w-w_ref)

[x_ref,w_ref] = jac.glquad(N,a=alpha,b=beta)
[x,w] = ps.jacobi.quad.glq(N,alpha=alpha,beta=beta)

print "Gauss-Lobatto quadrature comparison error is ", norm(x-x_ref) + norm(w-w_ref)

### Now see if it actually satisifies definitions of quadrature
[x,w] = ps.jacobi.quad.gq(N,alpha=alpha,beta=beta)
polys = ps.jacobi.eval.jpoly(x,range(1,2*N),alpha=alpha,beta=beta)
print "GQ quadrature error is ", norm(sum(polys.T*w, axis=1))

[x,w] = ps.jacobi.quad.grq(N,alpha=alpha,beta=beta)
polys = ps.jacobi.eval.jpoly(x,range(1,2*N-1),alpha=alpha,beta=beta)
print "GRQ quadrature error is ", norm(sum(polys.T*w, axis=1))

[x,w] = ps.jacobi.quad.glq(N,alpha=alpha,beta=beta)
polys = ps.jacobi.eval.jpoly(x,range(1,2*N-2),alpha=alpha,beta=beta)
print "GLQ quadrature error is ", norm(sum(polys.T*w, axis=1))
