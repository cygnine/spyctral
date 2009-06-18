#! /usr/bin/env python
# 
# Functions for evaluation and quadrature of Hermite polynomials
# Weight function is t**(mu/2)*exp(-t**2) CHECK

__all__ = []

import numpy as _np
import scipy.special as _nf
from spyctral import opoly1d

# Returns recurrence coefficients for certain values of n
def recurrence_ns(ns,mu=0,shift=0.,scale=1.):

    from numpy import arange
    from spyctral.opoly1d.eval import recurrence_scaleshift

    ns = _np.array(ns)
    N = ns.size
    ns = ns.reshape([N])

    a_s = _np.zeros([N])
    b_s = _np.zeros([N])

    b_s[ns==0] = _nf.gamma(mu+1/2.)
    b_s[ns>=1] = 1/2.*(arange(1,N))
    ks = range(1,N,2)
    b_s[ks] += mu
    
    return recurrence_scaleshift([a_s,b_s],scale=scale,shift=shift)

# Returns the first N recurrence coefficients for the Hermite polynomials
def recurrence(N,mu=0,shift=0.,scale=1.):

    from numpy import arange
    return recurrence_ns(arange(N),mu,shift,scale)

# Evaluates the monic Hermite polynomials of order mu at the locations x
def hpoly(x,n,mu=0.,d=0,shift=0.,scale=1.):

    N = _np.max(n)
    [a,b] = recurrence(N+1,mu,shift=shift,scale=scale)
    return opoly1d.eval_opoly(x,n,a,b,d)

# Evaluates the L2 normalized Hermite polynomials of order mu at the locations x
def hpolyn(x,n,mu=0.,d=0,shift=0.,scale=1.):

    N = _np.max(n)
    [a,b] = recurrence(N+2,mu,shift=shift,scale=scale)
    return opoly1d.eval_opolyn(x,n,a,b,d)

# Returns the N-point Hermite-Gauss quadrature rule
def gquad(N,mu=0.,shift=0.,scale=1.):

    [a_s,b_s] = recurrence(N,mu,shift=shift,scale=scale)
    return opoly1d.opoly_gq(a_s,b_s,N)

# Returns the N-point Jacobi-Gauss-Radau(mu) quadrature rule over the interval
# (-scale,scale)+shift
def grquad(N,mu=0.,r0=0.,shift=0,scale=1) : 

    [a_s,b_s] = recurrence(N,mu,shift=shift,scale=scale)
    return opoly1d.opoly_grq(a_s,b_s,N,r0=r0)

# Returns the weight function for the Hermite polynomials evaluated at a
# particular location
def weight(x,mu=0.,shift=0.,scale=1.):
    xt = (x-shift)/scale
    return _np.abs(xt)**(2*mu)*_np.exp(-xt**2)

# Returns the square root of the weight function for
# the Hermite polynomials evaluated at a particular location
def w_sqrt(x,mu=0.,shift=0.,scale=1.):
    xt = (x-shift)/scale
    w = _np.exp(-x**2/2)*_np.abs(x)**mu
    return w

# Returns the derivative of the square root of the weight function for
# the Hermite polynomials evaluated at a particular location
def dw_sqrt(x,mu=0.,shift=0.,scale=1.):
    xt = (x-shift)/scale
    w = _np.exp(-x**2/2)*(mu*_np.abs(x)**(mu-1) - x*_np.abs(x)**mu)
    return w/scale

########################################################
#                 HELPER FUNCTIONS                     #
########################################################

# Given a function, computes modes of the function in a certain class expansion
# using the Gauss quadrature native to that expansion. Computes the first N
# modes of f in the (mu) expansion using a Q-point (mu) Gauss quadrature
def expand_gq(f,N,Q,mu=0.):

    [x,w] = gquad(Q,mu)
    x = x.squeeze()
    w = w.squeeze()
    ps = hpolyn(x,range(N),mu)
    return _np.dot(ps.T,w*f(x))
