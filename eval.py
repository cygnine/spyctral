# Module for the evaluation of the rational Wiener functions

import numpy as _np
import scipy as _sp

import maps

__all__ = ['genwiener',
           'genwienerw']

# Evaluates the orthonormalized unweighted Wiener rational functions
def genwiener(x,k,s=1.,t=0.,shift=0.,scale=1.):

    from numpy import sqrt
    from fourier.genfourier import genfourier

    # Preprocessing and setup
    x = _np.array(x)
    x = x.ravel()
    k = _np.array(k,dtype='int')
    k = k.ravel()

    theta = maps.x2theta(x,shift=shift,scale=scale)

    return genfourier(theta,k,s-1.,t)/sqrt(scale)

# Evaluates the derivative of the orthonormalized unweighted Wiener rational
# functions
def dgenwiener(x,k,s=1.,t=0.):

    from fourier.genfourier import dgenfourier

    # Preprocessing and setup
    x = _np.array(x)
    x = x.ravel()
    k = _np.array(k,dtype='int')
    k = k.ravel()   

    theta = maps.x2theta(x,shift=shift,scale=scale)
    r = maps.x2r(x,shift=shift,scale=scale)

    return ((1+r)*(dgenfourier(theta,k,s-1.,t).T)).T/scale

# Evaluates the orthonormalized weighted Wiener rational functions
def genwienerw(x,k,s=1.,t=0.,shift=0.,scale=1.):

    from numpy import sqrt
    from fourier.genfourier import genfourier

    # Preprocessing and setup
    x = _np.array(x)
    x = x.ravel()
    k = _np.array(k,dtype='int')
    k = k.ravel()

    theta = maps.x2theta(x,shift=shift,scale=scale)

    psi = genfourier(theta,k,s-1.,t)

    psi = (wx_sqrt((x-shift)/scale,s,t)*(psi.T)).T

    return psi/sqrt(scale)

# Evaluates the orthonormalized weighted Wiener rational functions
def xiw(x,n,s=1.,t=0.,shift=0.,scale=1.):

    from numpy import sqrt
    from fourier.genfourier import genfourier

    # Preprocessing and setup
    x = _np.array(x)
    x = x.ravel()
    n = _np.array(n,dtype='int')
    n = n.ravel()

    theta = maps.x2theta(x,shift=shift,scale=scale)

    psi = genfourier(theta,n,s-1.,t).real

    psi = (wx_sqrt((x-shift)/scale,s,t)*(psi.T)).T

    psi[:,n==0] *= sqrt(2)
    psi[:,n!=0] *= 2

    return psi/sqrt(scale)

# Evaluates the derivative of the orthonormalized weighted Wiener rational functions
def dgenwienerw(x,k,s=1.,t=0.,shift=0.,scale=1.):

    from numpy import sqrt
    from fourier.genfourier import genfourier, dgenfourier
    from maps import dthetadx

    # Preprocessing and setup
    x = _np.array(x)
    x = x.ravel()
    k = _np.array(k,dtype='int')
    k = k.ravel()

    theta = maps.x2theta(x,shift=shift,scale=scale)

    # First term: wx_sqrt * d/dx Phi
    psi = (dthetadx(x,shift=shift,scale=scale)*dgenfourier(theta,k,s-1.,t).T).T
    psi = (wx_sqrt(x,s,t,shift=shift,scale=scale)*(psi.T)).T
    
    # Second term: d/dx wx_sqrt * Phi
    psi += (dwx_sqrt(x,s,t,shift=shift,scale=scale)*genfourier(theta,k,s-1.,t).T).T

    return psi/sqrt(scale)

# Evaluates the derivative of the orthonormalized weighted Wiener rational functions
def dxiw(x,n,s=1.,t=0.,shift=0.,scale=1.):

    from numpy import sqrt
    from fourier.genfourier import genfourier, dgenfourier
    from maps import dthetadx

    # Preprocessing and setup
    x = _np.array(x)
    x = x.ravel()
    n = _np.array(n,dtype='int')
    n = n.ravel()

    theta = maps.x2theta(x,shift=shift,scale=scale)

    # First term: wx_sqrt * d/dx Phi
    psi = (dthetadx(x,shift=shift,scale=scale)*dgenfourier(theta,n,s-1.,t).real.T).T
    psi = (wx_sqrt(x,s,t,shift=shift,scale=scale)*(psi.T)).T
    
    # Second term: d/dx wx_sqrt * Phi
    psi += (dwx_sqrt(x,s,t,shift=shift,scale=scale)*genfourier(theta,n,s-1.,t).real.T).T

    psi[:,n==0] *= sqrt(2)
    psi[:,n!=0] *= 2

    return psi/sqrt(scale)

# Evaluates the weight function
def wx(x,s=1.,t=0.,shift=0.,scale=1.):

    from fourier.genfourier import wtheta

    theta = maps.x2theta(x,shift=shift,scale=scale)

    y = (x-shift)/scale
    return 2*wtheta(theta,s-1,t)/(1+y**2)

# Evaluates the phase-shifted square root of the weight function
def wx_sqrt(x,s=1.,t=0.,shift=0.,scale=1.):

    y = (x-shift)/scale
    weightsqrt = y**t/(y-1j)**(s+t)
    weightsqrt *= 2**((s+t)/2.)
    return weightsqrt

# Evaluates the derivative of the phase-shifted square root of the weight
# function
def dwx_sqrt(x,s=1.,t=0.,shift=0.,scale=1.):

    y = (x-shift)/scale
    dws = t*(y-1j) - y*(s+t)
    dws *= (y**(t-1))/((y-1j)**(s+t+1.))
    dws *= 2**((s+t)/2.)
    return dws/scale
