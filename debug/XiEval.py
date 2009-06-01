#!/usr/bin/env python

# Script to test evaluation of XiW

import numpy as np
from numpy import exp
import wienerfun as wf

f = lambda x: exp(-x**2)
df = lambda x: -2*x*exp(-x**2)

ns = range(100)

N = 500

[x,w] = wf.quad.genwienerw_pgquad(2*N)
x = x[N:]
w = w[N:]

xis = wf.eval.xiw(x,ns)
modes = np.dot(xis.T.conj()*w,f(x))

mass = np.dot(xis.T.conj()*w,xis)

dxis = wf.eval.dxiw(x,ns)

stiff = np.dot(xis.T.conj()*w,dxis)
stiff[np.abs(stiff)<1e-10] = 0

dfx = np.dot(dxis,modes)

