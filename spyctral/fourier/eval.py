# !/usr/bin/env python
# 
# Routines for evaluation of Fourier functions

#__all__ = ['fseval']

from scipy import pi

def fseries(theta,k,gamma=0.,delta=0.,shift=0.,scale=1.):
    """
    Evaluates the generalized Szego-Fourier functions at the locations theta \in 
    [-pi,pi]. This function mods the inputs theta to lie in this interval and then
    evaluates them. The function class is (g,d), and the function index is the
    vector of integers k
    """

    from numpy import array, ndarray, abs, zeros
    from numpy import sin, cos, sqrt, sign
    from scipy import pi

    from spyctral.jacobi.eval import jpoly
    from spyctral.common.maps import standard_scaleshift as sss
    from spyctral.common.maps import physical_scaleshift as pss

    # Preprocessing: unravelling, etc.
    theta = array(theta)
    theta = theta.ravel()
    sss(theta,shift=shift,scale=scale)
    theta = (theta+pi) % (2*pi) - pi
    if type(k) != (ndarray or list):
        k = [k]
    k = array(k,dtype='int')
    k = k.ravel()
    kneq0 = k != 0

    r = cos(theta)

    # Evaluate polynomials and multiplication factors
    p1 = jpoly(r,abs(k),delta-1/2.,gamma-1/2.).reshape([theta.size,k.size])

    # Add things together
    Psi = zeros([theta.size,k.size],dtype=complex)
    Psi[:,~kneq0] = 1/sqrt(2)*p1[:,~kneq0]

    if k[kneq0].any():
        p2 = jpoly(r,abs(k[kneq0])-1,delta+1/2.,gamma+1/2.).\
                reshape([theta.size,k[kneq0].size])
        kmat = sign(k[kneq0])
        tmat = sin(theta)
        p2 = 1j*(p2.T*tmat).T*kmat
        Psi[:,kneq0] = 1/2.*(p1[:,kneq0] + p2)

    pss(theta,shift=shift,scale=scale)
    return Psi.squeeze()


def dfseries(theta,k,gamma=0.,delta=0.,shift=0.,scale=1.):
    """
    Evaluates the derivative of the generalized Szego-Fourier functions at the locations theta \in 
    [-pi,pi]. This function mods the inputs theta to lie in this interval and then
    evaluates them. The function class is (gamma,delta), and the function index is the
    vector of integers k
    """

    from numpy import array, zeros, dot, any, sum
    from numpy import sin, cos, sqrt, abs, sign
    from scipy import pi

    from spyctral.jacobi.eval import jpoly, djpoly
    from spyctral.common.maps import standard_scaleshift as sss
    from spyctral.common.maps import physical_scaleshift as pss

    # Preprocessing: unravelling, etc.
    theta = array(theta)
    theta = theta.ravel()
    sss(theta,shift=shift,scale=scale)
    # Now theta \in [-pi,pi]
    theta = (theta+pi) % (2*pi) - pi
    k = array(k)
    k = k.ravel()
    kneq0 = k != 0
    a = delta-1/2.
    b = gamma-1/2.

    r = cos(theta)

    # Term-by-term: first the even term
    dPsi = zeros([theta.size,k.size],dtype=complex)
    dPsi += 1/2.*djpoly(r,abs(k),a,b)
    dPsi = (-sin(theta)*dPsi.T).T
    dPsi[:,~kneq0] *= sqrt(2)

    # Now the odd term:
    if any(kneq0):
        term2 = zeros([theta.size,sum(kneq0)],dtype=complex)
        term2 += (cos(theta)*jpoly(r,abs(k[kneq0])-1,a+1,b+1).T).T
        term2 += (-sin(theta)**2*djpoly(r,abs(k[kneq0])-1,a+1,b+1).T).T
        term2 = 1./2*term2*(1j*sign(k[kneq0]))
    else:
        term2 = 0.

    dPsi[:,kneq0] += term2

    # Transform theta back to original interval
    pss(theta,shift=shift,scale=scale)
    return dPsi/scale

# Evaluates the generalized weighted Szego-Fourier functions at the locations theta \in 
# [-pi,pi]. This function mods the inputs theta to lie in this interval and then
# evaluates them. The function class is (g,d), and the function index is the
# vector of integers k
def weighted_fseries(theta,k,gamma=0.,delta=0.,shift=0.,scale=1.):

    from spyctral.common.maps import standard_scaleshift as sss
    from spyctral.common.maps import physical_scaleshift as pss
    from numpy import sqrt, array
    from weights import sqrt_weight_bias as wsqrt_bias

    theta = array(theta)
    theta = theta.ravel()
    sss(theta,shift=shift,scale=scale)

    psi = fseries(theta,k,gamma=gamma,delta=delta,shift=0.,scale=1.)
    phi = (wsqrt_bias(theta,gamma=gamma,delta=delta,shift=0.,scale=1.)*psi.T).T

    pss(theta,shift=shift,scale=scale)
    # Scaling:
    return phi/sqrt(scale)

# Evaluates the derivative of the shift/scaled weighted generalized
# Fourier functions. 
def dweighted_fseries(theta,ks,gamma=0.,delta=0.,shift=0.,scale=1.):
    from numpy import sqrt
    from weights import sqrt_weight_bias as wsqrt_bias
    from weights import dsqrt_weight_bias as dwsqrt_bias

    w = wsqrt_bias(theta,gamma=gamma,delta=delta,scale=scale,shift=shift)
    dw = dwsqrt_bias(theta,gamma=gamma,delta=delta,scale=scale,shift=shift)
    Psi = fseries(theta,ks,gamma=gamma,delta=delta,scale=scale,shift=shift)
    dPsi = dfseries(theta,ks,gamma=gamma,delta=delta,scale=scale,shift=shift)

    # yay product rule
    return (dw*Psi.T + w*dPsi.T).T/sqrt(scale)
