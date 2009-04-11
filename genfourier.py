# Module for defining the evaluation of the generalized Fourier functions

__all__ = ['genfourier',
           'genfourierw']

import numpy as _np
import scipy as _sp
from scipy import pi

import opoly1.jacobi as jac

# Evaluates the generalized Szego-Fourier functions at the locations theta \in 
# [-pi,pi]. This function mods the inputs theta to lie in this interval and then
# evaluates them. The function class is (g,d), and the function index is the
# vector of integers k
def genfourier(theta,k,g=0.,d=0.,shift=0.,scale=1.):
    from spectral_common import forward_scaleshift as fss
    from spectral_common import backward_scaleshift as bss

    # Preprocessing: unravelling, etc.
    theta = _np.array(theta)
    theta = theta.ravel()
    fss(theta,shift=shift,scale=scale)
    theta = (theta+pi) % (2*pi) - pi
    if type(k) != (_np.ndarray or list):
        k = [k]
    k = _np.array(k,dtype='int')
    k = k.ravel()
    kneq0 = k != 0

    r = _np.cos(theta)

    # Evaluate polynomials and multiplication factors
    p1 = jac.jpolyn(r,_np.abs(k),d-1/2.,g-1/2.).reshape([theta.size,k.size])
    #p2 = jac.jpolyn(r,_np.abs(k[kneq0])-1,d+1/2.,g+1/2.)
    #kmat = _np.diag(_np.sign(k[kneq0]))
    #tmat = _np.diag(_np.sin(theta))
    #p2 = 1j*_np.dot(_np.dot(tmat,p2),kmat)

    # Add things together
    Psi = _np.zeros([theta.size,k.size],dtype='complex128')
    Psi[:,~kneq0] = 1/_np.sqrt(2)*p1[:,~kneq0]
    #Psi[:,kneq0] = 1./2.*(p1[:,kneq0] + p2)

    if k[kneq0].any():
        p2 = jac.jpolyn(r,_np.abs(k[kneq0])-1,d+1/2.,g+1/2.).\
                reshape([theta.size,k[kneq0].size])
        #kmat = _np.diag(_np.sign(k[kneq0]))
        kmat = _np.sign(k[kneq0])
        #tmat = _np.diag(_np.sin(theta))
        tmat = _np.sin(theta)
        p2 = 1j*(p2.T*tmat).T*kmat
        #p2 = 1j*_np.dot(_np.dot(tmat,p2),kmat)
        Psi[:,kneq0] = 1./2.*(p1[:,kneq0] + p2)

    # put theta back to original state
    bss(theta,shift=shift,scale=scale)
    return Psi.squeeze()

# Evaluates the derivative of the generalized Szego-Fourier functions at the locations theta \in 
# [-pi,pi]. This function mods the inputs theta to lie in this interval and then
# evaluates them. The function class is (g,d), and the function index is the
# vector of integers k
def dgenfourier(theta,k,g=0.,d=0.,shift=0.,scale=1.):
    from spectral_common import forward_scaleshift as fss
    from spectral_common import backward_scaleshift as bss

    # Preprocessing: unravelling, etc.
    theta = _np.array(theta)
    theta = theta.ravel()
    fss(theta,shift=shift,scale=scale)
    # Now theta \in [-pi,pi]
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

    # Now the odd term:
    if _np.any(kneq0):
        term2 = _np.zeros([theta.size,_np.sum(kneq0)],dtype='complex128')
        term2 += _np.dot(_np.diag(_np.cos(theta)),jac.jpolyn(r,_np.abs(k[kneq0])-1,a+1,b+1))
        ####term2 += _np.dot(_np.diag(-_np.sin(theta)**2),jac.jpolyn(r,_np.abs(k[kneq0])-1,a+1,b+1,d=1))
        term2 += _np.dot(_np.diag(-_np.sin(theta)**2),jac.djpolyn(r,_np.abs(k[kneq0])-1,a+1,b+1))
        term2 = 1./2*_np.dot(term2,_np.diag(1j*_np.sign(k[kneq0])))
    else:
        term2 = 0.

    dPsi[:,kneq0] += term2

    # Transform theta back to original interval
    bss(theta,shift=shift,scale=scale)
    return dPsi/scale

# Evaluates the generalized weighted Szego-Fourier functions at the locations theta \in 
# [-pi,pi]. This function mods the inputs theta to lie in this interval and then
# evaluates them. The function class is (g,d), and the function index is the
# vector of integers k
def genfourierw(theta,k,g=0.,d=0.,shift=0.,scale=1.):
    from spectral_common import forward_scaleshift as fss
    from spectral_common import backward_scaleshift as bss
    from numpy import sqrt

    theta = _np.array(theta)
    theta = theta.ravel()
    fss(theta,shift=shift,scale=scale)

    psi = genfourier(theta,k,g,d,shift=0,scale=1)

    #w = (1-_np.cos(theta))**(d/2.)
    #w *= (1+_np.cos(theta))**(g/2.)

    #phi = _np.dot(_np.diag(w),psi)
    phi = _np.dot(_np.diag(wtheta_sqrt(theta,g=g,d=d,shift=0.,scale=1.)),psi).squeeze()

    bss(theta,shift=shift,scale=scale)
    # Scaling:
    return phi/sqrt(scale)

# Evaluates the derivative of the shift/scaled weighted generalized
# Fourier functions. 
def dgenfourierw(theta,ks,g=0.,d=0.,shift=0.,scale=1.):
    from numpy import sqrt

    w = wtheta_sqrt(theta,g=g,d=d,scale=scale,shift=shift)
    dw = dwtheta_sqrt(theta,g=g,d=d,scale=scale,shift=shift)
    Psi = genfourier(theta,ks,g=g,d=d,scale=scale,shift=shift)
    dPsi = dgenfourier(theta,ks,g=g,d=d,scale=scale,shift=shift)

    return (dw*Psi.T + w*dPsi.T).T/sqrt(scale)

# Defines the regular square root of the Szego-Fourier weight
def wtheta(theta,g=0.,d=0.,shift=0.,scale=1.):
    from numpy import array
    from spectral_common import forward_scaleshift as fss
    from spectral_common import backward_scaleshift as bss

    theta = array(theta)
    fss(theta,scale=scale,shift=shift)
    w = ((1-_np.cos(theta))**d)*((1+_np.cos(theta))**g)
    bss(theta,scale=scale,shift=shift)
    return w

# Defines the conjugate-biased weight function for the Szego-Fourier basis sets
def wtheta_sqrt(theta,g=0.,d=0.,shift=0.,scale=1.):
    from spectral_common import forward_scaleshift as fss
    from spectral_common import backward_scaleshift as bss
    from scipy import power as pw
    from numpy import exp, sin, cos

    fss(theta,scale=scale,shift=shift)
    phase = exp(1j*(g+d)/2.*(pi-theta))
    #w = phase*(_np.sin(theta/2.)**d)*(_np.cos(theta/2.)**g)*2**((g+d)/2.)
    w = phase*( pw(sin(theta/2.),d) * \
                pw(cos(theta/2.),g)) *\
              2**((g+d)/2.)
    bss(theta,scale=scale,shift=shift)

    return w

# Defines the derivative of the wtheta_sqrt function
def dwtheta_sqrt(theta,g=0.,d=0.,shift=0.,scale=1.):
    from spectral_common import forward_scaleshift as fss
    from spectral_common import backward_scaleshift as bss
    from numpy import sin,cos,exp

    fss(theta,scale=scale,shift=shift)
    phase = exp(1j*(g+d)/2.*(pi-theta))*\
            2**((g+d-4)/2.)*\
            sin(theta/2.)**(d-1)*\
            cos(theta/2.)**(g-1)
    w = phase*( d*(1+exp(-1j*theta)) -\
                g*(1+exp(1j*theta)))
    bss(theta,scale=scale,shift=shift)

    return w/scale
