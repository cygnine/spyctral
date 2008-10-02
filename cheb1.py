# !/usr/bin/env python
#
# Module for the Chebyshev polynomials: extensively uses the Jacobi module

__all__ = ['poly', 'polyn', 'gquad', 'grquad', 'glquad']
import jacobi
from numpy import ones
from scipy import pi

# Returns the size-N recurrence vectors [a,b] for the Chebyshev polynomials of
# the first kind
def recurrence(N,shift=0.,scale=1.) :
    return jacobi.recurrence(N,alpha=-1/2.,beta=-1/2.,shift=shift,scale=scale)

# Returns the evaluations of the monic Chebyshev polynomials of the first kind,
# order n (list), evaluated at the points x
def poly(x,n,shift=0.,scale=1.) :
    return jacobi.jpoly(x,n,alpha=-1/2.,beta=-1/2.,shift=shift,scale=scale)

# Returns the evaluations of the L^2 normalized Chebyshev polynomials of the first kind,
# order n (list), evaluated at the points x
def polyn(x,n,shift=0.,scale=1.) :
    return jacobi.jpolyn(x,n,alpha=-1/2.,beta=-1/2.,shift=shift,scale=scale)

# Returns the N-point Chebyshev-Gauss quadrature shifted to the interval
# (-scale, scale) + shift
def gquad(N,shift=0.,scale=1.) :

    w = scale*pi/float(N)*ones([N,1])

    temp = linspace(pi,0,N+1)
    x = scale*cos(temp[:N]-pi/(2.*N)) + shift

    return [x,w]
