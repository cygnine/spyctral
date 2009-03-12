#! /usr/bin/env python
# submodule for Fourier quadrature
#
# 20081029 -- acn

__all__ = ['gquad',
           'genfourier_gquad']

import numpy as _np
from scipy import pi

# Returns the default Fourier quadrature rule over the interval (-pi*scale,
# pi*scale) + shift
def gquad(N,scale=0.,shift=0.) :
    
    x = _np.linspace(-pi,pi,N+1)
    x = x[:N] + 1./2*(x[1]-x[0])

    w = 2*pi/N*_np.ones(x.shape)

    return [x,w]

# Returns generalized Szego-Fourier qudarature rules over the interval (-pi,pi).
# The rules are constructed for the unweighted functions, and with the intent
# that they will eventually be mapped to the real line so that we try not to
# place nodes at x=\pm\pi
def genfourier_gquad(N,g=0.,d=0.):

    from opoly1 import jacobi as jac

    N = int(N)
    tol = 1e-8
    #if (_np.abs(g)<tol)&(_np.abs(d)<tol):
    #    return gquad(N)

    if (N%2)==0:

        [r,wr] = jac.gquad(N/2,d-1/2.,g-1/2.)
        r = r.squeeze()
        wr = wr.squeeze()
        temp = _np.arccos(r[::-1])
        wr = wr[::-1]
        theta = _np.hstack((-temp[::-1],temp))
        w = _np.hstack((wr[::-1],wr))
        return [theta,w]

    else:

        [r,wr] = jac.grquad((N+1)/2,d-1/2.,g-1/2.,r0=1.)
        r = r.squeeze()
        wr = wr.squeeze()
        temp = _np.arccos(r[::-1])
        # Silly arccos machine epsilson crap
        temp[_np.isnan(temp)] = 0.
        theta = _np.hstack((-temp[::-1],temp[1:]))
        wr = wr[::-1]
        wr[0] *= 2
        w = _np.hstack((wr[::-1],wr[1:]))
        return [theta,w]

# Returns the Szego-Fourier quadrature for the weighted functions
def genfourierw_pgquad(N,g=0.,d=0.):

    from fourier.genfourier import wtheta
    [theta,w] = genfourier_gquad(N,g,d)
    return [theta,w/wtheta(theta,g,d)]

# Returns the modes of the input numpy array along the axis 0 assuming that the
# nodal values are at the Szego-Fourier quadrature points. Returns modes for the
# unweighted Szego-Fourier functions.
# The FFT is only usable if G and D are integers; thus, this function casts them
# as thus
def fft(fx,G=0,D=0):
    from fourier.connection import int_connection
    return int_connection(fft_base(fx),0.,0.,int(G),int(D))

# Returns modes of input num array along axis 0 assuming that the nodal values
# are at the Szego-Fourier quadrature points, and that the nodal basis functions 
# Returns modes for the weighted Szego-Fourier functions.
def wfft(theta,fx,G=0,D=0):
    from fourier.genfourier import wtheta_sqrt as w
    G = int(G)
    D = int(D)

    wx = w(theta,G,D)
    return fft(fx/wx,G,D)

# Base FFT Routine: does phase shifts, etc for using canonical FFT for
# orthonormalized canoncial Fourier basis on Szego-Fourier quadrature points
def fft_base(fx):

    from numpy import sqrt, exp
    from numpy.fft import fft

    fx = fx.squeeze()
    N = fx.shape[0]
    shift = -_np.array(range(N))*(1+1./N)
    shift = exp(1j*pi*shift.T)*sqrt(2*pi)/N

    #FX = fft(fx,axis=0)*shift
    FX = _np.dot(_np.diag(shift),fft(fx,axis=0))
    shifts = (-1)**(N+1)
    temp = (N+1)/2
    return _np.hstack((shifts*FX[temp:],FX[:temp]))
