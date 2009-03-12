# Module for determining quadrature stuff for generalized Wiener rational
# functions

import numpy as _np

import maps

__all__ = ['genwiener_gquad',
           'genwienerw_pgquad',
           'fft',
           'wfft']

def quad(N,s=1.,t=0.,shift=0.,scale=1.):
    return genwienerw_pgquad(N,s=s,t=t,shift=shift,scale=scale)

# Returns N k-indices, using the default standard bias (to the
# left/negative if N is even)
def N_to_ks(N):

    from numpy import arange

    N = int(N)
    if bool(N%2):
        N = (N-1)/2
        ks = arange(-N,N+1)
    else:
        N = N/2
        ks = arange(-N,N)

    return ks

# Returns the `Gauss' quadrature formula for the unweighted functions:
def genwiener_gquad(N,s=1.,t=0,shift=0.,scale=1.):
    from fourier.quad import genfourier_gquad as gq

    [theta,w] = gq(N,s-1.,t)
    x = maps.theta2x(theta,shift=shift,scale=scale)
    w *= scale

    return [x,w]

# Returns the `Gauss' quadrature rule for the weighted functions:
def genwienerw_pgquad(N,s=1.,t=0,shift=0.,scale=1.):
    from fourier.quad import genfourierw_pgquad as gq
    from wienerfun.eval import wx

    [theta,w] = gq(N,s-1.,t)
    x = maps.theta2x(theta,shift=shift,scale=scale)
    w *= scale
    w /= wx(x,s,t,shift=shift,scale=scale)

    return [x,w]

# Performs the FFT on nodal evaluations to recover the modal coefficients for
# the unweighted Wiener functions. Assuming the nodal evaluations are located at
# the direct map of the canonical Szego-Fourier quadrature points.
def fft(fx,s=1.,t=0.):

    from fourier.quad import fft as fft_theta

    return fft_theta(fx,s-1.,t)

# Performs the FFT on nodal evaluations to recover the modal coefficients for
# the weighted Wiener functions. Assuming the nodal evaluations are located at
# the direct map of the canonical Szego-Fourier quadrature points.
def wfft(x,fx,s=1.,t=0.):

    from fourier.quad import wfft as wfft_theta

    return wfft_theta(maps.x2theta(x),fx,s-1.,t)
