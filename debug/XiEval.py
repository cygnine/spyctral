#!/usr/bin/env python

# Script to test evaluation of XiW

import numpy as np
import wienerfun as wf

ns = range(100)

N = 500

[x,w] = wf.quad.genwienerw_pgquad(2*N)
x = x[N:]
w = w[N:]

xis = wf.eval.xiw(x,ns)

mass = np.dot(xis.T.conj()*w,xis)

dxis = wf.eval.dxiw(x,ns)

stiff = np.dot(xis.T.conj()*w,dxis)
stiff[np.abs(stiff)<1e-10] = 0
