# !/usr/bin/env python
#
# Module for the Chebyshev polynomials: extensively uses the Jacobi module

__all__ = ['poly', 'polyn', 'gquad', 'grquad', 'glquad','chebfft']
import jacobi
from numpy import ones, linspace, cos, vstack, array, matrix, exp, sqrt
from numpy.fft import rfft
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

    w = scale*pi/float(N)*ones([N])

    temp = linspace(pi,0,N+1)
    x = scale*cos(temp[:N]-pi/(2.*N)) + shift

    return [x, w]

# Returns the N-point Chebyshev-Gauss-Lobatto quadrature shifted to the interval
# (-scale,scale)+shift
def glquad(N,shift=0.,scale=1.) :

    w = scale*pi/float(N-1)*ones([N])
    w[0] *= 1/2.
    w[N-1] *= 1/2.

    temp = linspace(pi,0,N)
    x = scale*cos(temp[:N]) + shift

    return [x,w]

# Returns the N-point Chebyshev-Gauss-Radau quadrature shifted to the interval
# (-scale,scale)+shift. The default fixed point is at r0 = -1. NOTE: This cannot
# be used if abs(r0) != 1. If the fixed point is in the interior of the
# interval, use jacobi.grquad instead.
def grquad(N,r0=-1.,shift=0.,scale=1.) :

    w = scale*pi/float(N-0.5)*ones([N])
    w[0] *= 1/2.

    temp = linspace(pi,-pi,2*N)
    x = scale*cos(temp[:N]) + shift

    if r0>0 :
        x = -x[N::-1]
        w = w[N::-1]

    return [x,w]

# Uses the FFT to compute the Chebyshev modes for the normalized polynomials
# for nodal points for the Chebyshev-Gauss type. Only works along rows right now
def chebfft(f) :
    N = f.shape[0]

    f = (rfft(vstack((f,f[::-1])),axis=0)/(2.*N))[0:N]

    shift = -array([range(N)])*(1+1/(2.*N))
    shift = sqrt(2*pi)*exp(1j*pi*shift.T)
    shift[0] = shift[0]/sqrt(2)

    f = matrix(array(f)*shift)

    return f.real
