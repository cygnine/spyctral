import numpy as np
from numpy import sin, cos, exp, dot
from numpy.linalg import norm
import scipy as sp
from scipy import rand, randn

import spyctral.jacobi as jac

N = 100
Ne = 200
f = lambda x: sin(x)

def shiftstuff(flag):
    if flag:
        alpha = -1/2. + int(10*rand())
        beta = -1/2. + int(10*rand())
        scale = 5*rand()
        shift = 5*randn()
    else:
        alpha = -1/2.
        beta = -1/2.
        scale = 1.
        shift = 0.
    keys = ['alpha', 'beta','scale','shift']
    return dict(zip(keys,[alpha,beta,scale,shift]))

params = shiftstuff(True)
A = params['alpha'] + 1/2.
B = params['beta'] + 1/2.
shift = params['shift']
scale = params['scale']

[r,w] = jac.quad.gq(N,alpha=-1/2.,beta=-1/2.,scale=scale,shift=shift)
[re,we] = jac.quad.gq(Ne,alpha=-1/2.,beta=-1/2.,scale=scale,shift=shift)
ps = jac.eval.jpoly(r,range(N),**params)
pse = jac.eval.jpoly(re,range(N),**params)
weight = jac.weights.weight(r,alpha=A,beta=B,shift=shift,scale=scale)

fr = f(r)
modes = dot(ps.T*w*weight,fr)
modes_fft = jac.jfft.jacfft(fr,A=A,B=B,scale=scale,shift=shift)
fft_overhead = jac.jfft.jacfft_overhead(N,A=A,B=B,scale=scale,shift=shift)
modes_fft2 = jac.jfft.jacfft_online(fr,fft_overhead)

print "JFFT error", norm(modes[:-(A+B)]-modes_fft[:-(A+B)])
print "JFFT on/off error", norm(modes-modes_fft2)

nodes_fft = jac.jfft.jacifft(modes_fft,A=A,B=B,scale=scale,shift=shift)
nodes_fft2 = jac.jfft.jacifft_online(modes_fft,fft_overhead)

print "JFFT nodal error", norm(nodes_fft - fr)
print "JFFT nodal on/off error", norm(nodes_fft2-fr)
