#!/usr/bin/env python
"""
* File Name : quad.py

* Creation Date : 2009-06-17

* Created By : Akil Narayan

* Last Modified : Wed 17 Jun 2009 04:29:25 PM EDT

* Purpose : Quadrature debugging
"""

import numpy as np
from numpy import sum, dot, eye
from numpy.linalg import norm
import scipy as sp
from scipy import rand, randn
import spyctral as ps
import opoly1.jacobi as jac

N = 100
L = 10*rand()
#L = 1.
a = 10*randn()
#a = 0.
alpha = -1/2. + 11*rand();
beta = -1/2. + 11*rand();

#### First compare against established opoly1 module
[x_ref,w_ref] = jac.gquad(N,a=alpha,b=beta,scale=L,shift=a)
[x,w] = ps.jacobi.quad.gq(N,alpha=alpha,beta=beta,scale=L,shift=a)

print "Gauss quadrature comparison error is ", norm(x-x_ref) + norm(w-w_ref)

[x_ref,w_ref] = jac.grquad(N,a=alpha,b=beta,scale=L,shift=a)
[x,w] = ps.jacobi.quad.grq(N,alpha=alpha,beta=beta,scale=L,shift=a)

print "Gauss-Radau quadrature comparison error is ", norm(x-x_ref) + norm(w-w_ref)

[x_ref,w_ref] = jac.glquad(N,a=alpha,b=beta,scale=L,shift=a)
[x,w] = ps.jacobi.quad.glq(N,alpha=alpha,beta=beta,scale=L,shift=a)

print "Gauss-Lobatto quadrature comparison error is ", norm(x-x_ref) + norm(w-w_ref)

### Now see if it actually satisifies definitions of quadrature
[x,w] = ps.jacobi.quad.gq(N,alpha=alpha,beta=beta,scale=L,shift=a)
polys = ps.jacobi.eval.jpoly(x,range(1,2*N),alpha=alpha,beta=beta,scale=L,shift=a)
print "GQ quadrature error is ", norm(sum(polys.T*w, axis=1))
polys = ps.jacobi.eval.jpoly(x,range(N),alpha=alpha,beta=beta,scale=L,shift=a)
mass = dot(polys.T*w,polys)

[x,w] = ps.jacobi.quad.grq(N,alpha=alpha,beta=beta,scale=L,shift=a)
polys = ps.jacobi.eval.jpoly(x,range(1,2*N-1),alpha=alpha,beta=beta,shift=a,scale=L)
print "GRQ quadrature error is ", norm(sum(polys.T*w, axis=1))

[x,w] = ps.jacobi.quad.glq(N,alpha=alpha,beta=beta,scale=L,shift=a)
polys = ps.jacobi.eval.jpoly(x,range(1,2*N-2),alpha=alpha,beta=beta,scale=L,shift=a)
print "GLQ quadrature error is ", norm(sum(polys.T*w, axis=1))

print "mass matrix error is", norm(mass - eye(N))
