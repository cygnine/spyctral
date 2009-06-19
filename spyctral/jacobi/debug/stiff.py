# Testing stiffness matrices

import spyctral.jacobi as jac
import numpy as np
from numpy import exp, sin, dot, cos
from numpy.linalg import norm
import scipy as sp
from scipy import rand,randn

f = lambda x: sin(x)
df = lambda x: cos(x)
N = 100
alpha = -1/2. + 10*rand()
beta = -1/2. + 10*rand()
scale = 1.
shift = 0.
params = {'alpha':alpha, 
          'beta': beta,
          'scale':scale,
          'shift':shift}

[r,w] = jac.quad.gq(N,**params)
fr = f(r)
ps = jac.eval.jpoly(r,range(N),**params)
dps = jac.eval.djpoly(r,range(N),**params)
modes = dot(ps.T*w,fr)
dmodes = jac.mats.stiff_apply(modes,alpha=alpha,beta=beta,scale=scale)
stiff_overhead = jac.mats.stiff_overhead(N,alpha=alpha,beta=beta,scale=scale)
dmodes2 = jac.mats.stiff_online(modes,stiff_overhead)

drec = dot(dps,modes)
drec_stiff = dot(ps,dmodes)
drec_stiff_offon = dot(ps,dmodes2)

print "Direct derivative error is ", norm(drec-df(r))
print "Stiffness derivative error is ", norm(drec_stiff-df(r))
print "Stiffness on/off derivative error is ", norm(drec_stiff_offon-df(r))
