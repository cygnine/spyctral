#! /usr/bin/env python
# submodule for Fourier quadrature
#
# 20081029 -- acn

__all__ = ['gquad',
           'genfourier_gquad']

# Returns generalized Szego-Fourier qudarature rules over the interval (-pi,pi).
# The rules are constructed for the unweighted functions, and with the intent
# that they will eventually be mapped to the real line so that we try not to
# place nodes at x=\pm\pi
def gq(N,gamma=0.,delta=0.,shift=0.,scale=1.):

    from numpy import arccos, hstack, isnan
    from spyctral.jacobi.quad import gq as jgq
    from spyctral.jacobi.quad import grq as jgrq
    from spyctral.common.maps import physical_scaleshift as pss

    N = int(N)
    tol = 1e-8

    if (N%2)==0:

        [r,wr] = jgq(N/2,delta-1/2.,gamma-1/2.)
        r = r.squeeze()
        wr = wr.squeeze()
        temp = arccos(r[::-1])
        wr = wr[::-1]
        theta = hstack((-temp[::-1],temp))
        w = hstack((wr[::-1],wr))
        pss(theta,scale=scale,shift=shift)
        return [theta,w]

    else:

        [r,wr] = jgrq((N+1)/2,delta-1/2.,gamma-1/2.,r0=1.)
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
def pgq(N,gamma=0.,delta=0.,shift=0.,scale=1.):

    from weights import weight
    [theta,w] = gq(N,gamma,delta,shift=shift,scale=scale)
    return [theta,w/weight(theta,gamma,delta,shift=shift,scale=scale)*scale]
