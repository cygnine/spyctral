# !/usr/bin/env python
# 
# Routines for evaluation of Fourier functions

__all__ = ['fseval']

import numpy as _np
from scipy import pi

# Evaluates the orthonormalized Fourier Series basis functions over the interval
# [-pi*scale, pi*scale]+shift. 
def fseval(x,ns,scale=1.,shift=0.) :

    ns = _np.array(ns)
    # Pre-processing:
    x = _np.array(x).ravel()
    X = x.size
    N = ns.size
    fs = _np.zeros((X,N),dtype=complex)
    count = 0

    for n in ns :
        fs[:,count] = 1./_np.sqrt(2*pi*scale)*_np.exp(1j*n*(x-shift)/scale)
        count += 1

    return fs
