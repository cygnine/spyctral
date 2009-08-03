#!/usr/bin/env python
"""
* File Name : quad.py

* Creation Date : 2009-06-17

* Created By : Akil Narayan

* Last Modified : Wed 17 Jun 2009 03:54:46 PM EDT

* Purpose :
"""
from numpy import ones,cos,linspace
from scipy import pi

def gq(N,scale=1.,shift=0.):

    #w = scale*pi/float(N)*ones([N])
    w = pi/float(N)*ones([N])

    temp = linspace(pi,0,N+1)
    x = scale*cos(temp[:N]-pi/(2.*N)) + shift

    return [x, w]

def glq(N,shift=0.,scale=1.) :

    #w = scale*pi/float(N-1)*ones([N])
    w = pi/float(N-1)*ones([N])
    w[0] *= 1/2.
    w[N-1] *= 1/2.

    temp = linspace(pi,0,N)
    x = scale*cos(temp[:N]) + shift

    return [x,w]


def grq(N,r0=-1.,shift=0.,scale=1.) :
    """
    Returns the N-point Chebyshev-Gauss-Radau quadrature shifted to the interval
    (-scale,scale)+shift. The default fixed point is at r0 = -1. NOTE: This cannot
    be used if abs(r0) is not 1. If the fixed point is in the interior of the
    interval, use jacobi.grquad instead.
    """

    #w = scale*pi/float(N-0.5)*ones([N])
    w = pi/float(N-0.5)*ones([N])
    w[0] *= 1/2.

    temp = linspace(pi,-pi,2*N)
    x = scale*cos(temp[:N]) + shift

    if r0>0 :
        x = -x[N::-1]
        w = w[N::-1]

    return [x,w]
