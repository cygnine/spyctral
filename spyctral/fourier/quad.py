#! /usr/bin/env python
# submodule for Fourier quadrature
#
# 20081029 -- acn

__all__ = ['gquad',
           'genfourier_gquad']

"""
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

# Returns the default Fourier quadrature rule over the interval (-pi*scale,
# pi*scale) + shift
def gquad(N,scale=0.,shift=0.) :
    
    x = _np.linspace(-pi,pi,N+1)
    x = x[:N] + 1./2*(x[1]-x[0])

    w = 2*pi/N*_np.ones(x.shape)

    return [x,w]
"""

# Returns generalized Szego-Fourier qudarature rules over the interval (-pi,pi).
# The rules are constructed for the unweighted functions, and with the intent
# that they will eventually be mapped to the real line so that we try not to
# place nodes at x=\pm\pi
def gq(N,g=0.,d=0.,shift=0.,scale=1.):

    from numpy import arccos, hstack, isnan
    from pyspec.jacobi.quad import gq as jgq
    from pyspec.jacobi.quad import grq as jgrq
    from pyspec.common.maps import physical_scaleshift as pss

    N = int(N)
    tol = 1e-8

    if (N%2)==0:

        [r,wr] = jgq(N/2,d-1/2.,g-1/2.)
        r = r.squeeze()
        wr = wr.squeeze()
        temp = arccos(r[::-1])
        wr = wr[::-1]
        theta = hstack((-temp[::-1],temp))
        w = hstack((wr[::-1],wr))
        pss(theta,scale=scale,shift=shift)
        return [theta,w]

    else:

        [r,wr] = jgrq((N+1)/2,d-1/2.,g-1/2.,r0=1.)
        r = r.squeeze()
        wr = wr.squeeze()
        temp = arccos(r[::-1])
        # Silly arccos machine epsilon crap
        temp[isnan(temp)] = 0.
        theta = hstack((-temp[::-1],temp[1:]))
        wr = wr[::-1]
        wr[0] *= 2
        w = hstack((wr[::-1],wr[1:]))
        pss(theta,scale=scale,shift=shift)
        return [theta,w]

# Returns the Szego-Fourier quadrature for the weighted functions
def pgq(N,g=0.,d=0.,shift=0.,scale=1.):

    from weights import w as weight
    [theta,w] = gq(N,g,d,shift=shift,scale=scale)
    return [theta,w/weight(theta,g,d,shift=shift,scale=scale)*scale]
