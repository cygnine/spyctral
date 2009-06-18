# Module for defining the evaluation of the generalized Fourier functions
# THIS IS A TEST MODULE: USED FOR DETERMINING NORMALIZATION CONSTANTS 
__all__ = ['genfourier',
           'genfourierw']

import numpy as _np
import scipy as _sp
from scipy import pi

import opoly1.jacobi as jac

# Defines normalization coefficients for the Szego-Fourier functions
def ck(k,g=0.,d=0.):
    return _np.ones(k.size)

def dk(k,g=0.,d=0.):
    return _np.ones(k.size)

# Evaluates the generalized Szego-Fourier functions at the locations theta \in 
# [-pi,pi]. This function mods the inputs theta to lie in this interval and then
# evaluates them. The function class is (g,d), and the function index is the
# vector of integers k
def genfourier(theta,k,g=0.,d=0.):

    # Preprocessing: unravelling, etc.
    theta = _np.array(theta)
    theta = theta.ravel()
    theta = (theta+pi) % (2*pi) - pi
    k = _np.array(k,dtype='int')
    k = k.ravel()
    kneq0 = k != 0

    r = _np.cos(theta)

    # Evaluate polynomials and multiplication factors
    p1 = jac.jpolyn(r,_np.abs(k),d-1/2.,g-1/2.)
    p1 *= ck(k)

    # Add things together
    Psi = _np.zeros([theta.size,k.size],dtype='complex128')
    Psi[:,~kneq0] = 1/_np.sqrt(2)*p1[:,~kneq0]

    if k[kneq0].any():
        p2 = jac.jpolyn(r,_np.abs(k[kneq0])-1,d+1/2.,g+1/2.)
        kmat = _np.diag(_np.sign(k[kneq0]))
        tmat = _np.diag(_np.sin(theta))
        p2 = 1j*_np.dot(_np.dot(tmat,p2),kmat)
        p2 *= dk(k[kneq0])
        Psi[:,kneq0] = 1./2.*(p1[:,kneq0] + p2)

    return Psi

# Evaluates the derivative of the generalized Szego-Fourier functions at the locations theta \in 
# [-pi,pi]. This function mods the inputs theta to lie in this interval and then
# evaluates them. The function class is (g,d), and the function index is the
# vector of integers k
def dgenfourier(theta,k,g=0.,d=0.):

    # Preprocessing: unravelling, etc.
    theta = _np.array(theta)
    theta = theta.ravel()
    theta = (theta+pi) % (2*pi) - pi
    k = _np.array(k)
    k = k.ravel()
    kneq0 = k != 0
    a = d-1/2.
    b = g-1/2.

    r = _np.cos(theta)

    # Term-by-term: first the even term
    dPsi = _np.zeros([theta.size,k.size],dtype='complex128')
    #dPsi += 1/2.*jac.jpolyn(r,_np.abs(k),a,b,d=1)
    dPsi += 1/2.*jac.djpolyn(r,_np.abs(k),a,b)
    dPsi = _np.dot(_np.diag(-_np.sin(theta)),dPsi)
    dPsi[:,~kneq0] *= _np.sqrt(2)
    dPsi *= ck(k)

    # Now the odd term:
    if _np.any(kneq0):
        term2 = _np.zeros([theta.size,_np.sum(kneq0)],dtype='complex128')
        term2 += _np.dot(_np.diag(_np.cos(theta)),jac.jpolyn(r,_np.abs(k[kneq0])-1,a+1,b+1))
        ####term2 += _np.dot(_np.diag(-_np.sin(theta)**2),jac.jpolyn(r,_np.abs(k[kneq0])-1,a+1,b+1,d=1))
        term2 += _np.dot(_np.diag(-_np.sin(theta)**2),jac.djpolyn(r,_np.abs(k[kneq0])-1,a+1,b+1))
        term2 = 1./2*_np.dot(term2,_np.diag(1j*_np.sign(k[kneq0])))
        term2 *= dk(k[kneq0])
    else:
        term2 = 0.

    dPsi[:,kneq0] += term2

    return dPsi

# Evaluates the generalized weighted Szego-Fourier functions at the locations theta \in 
# [-pi,pi]. This function mods the inputs theta to lie in this interval and then
# evaluates them. The function class is (g,d), and the function index is the
# vector of integers k
def genfourierw(theta,k,g=0.,d=0.):

    theta = _np.array(theta)
    theta = theta.ravel()

    psi = genfourier(theta,k,g,d)

    w = (1-_np.cos(theta))**(d/2.)
    w *= (1+_np.cos(theta))**(g/2.)

    #phi = _np.dot(_np.diag(w),psi)
    phi = _np.dot(_np.diag(wtheta_sqrt(theta,g,d)),psi).squeeze()

    return phi

# Defines the regular square root of the Szego-Fourier weight
def wtheta(theta,g=0.,d=0.):
    return ((1-_np.cos(theta))**d)*((1+_np.cos(theta))**g)

# Defines the conjugate-biased weight function for the Szego-Fourier basis sets
def wtheta_sqrt(theta,g=0.,d=0.):

    phase = _np.exp(1j*(g+d)/2.*(pi-theta))

    return phase*(_np.sin(theta/2.)**d)*(_np.cos(theta/2.)**g)*2**((g+d)/2.)
