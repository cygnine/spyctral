#! /usr/bin/env python
# 
# Functions for evaluation and quadrature of Laguerre polynomials
# Weight function is exp(-t)

__all__ = []

import numpy as _np
import scipy.special as _nf
import opoly1

# Returns recurrence coefficients for certain values of n
def recurrence_ns(ns,shift=0.,scale=1.):

    from numpy import arange
    from opoly1.eval import recurrence_scaleshift

    ns = _np.array(ns,dtype=float)
    N = ns.size
    ns = ns.reshape([N])

    a_s = _np.zeros([N])
    b_s = _np.zeros([N])
    b_s = ns**2
    b_s[ns==0] = 1

    a_s = 2*ns + 1
    #b_s[ns==0] = 1
    #b_s[ns>=1] = ns**2;
    
    return recurrence_scaleshift([a_s,b_s],scale=scale,shift=shift)

# Returns the first N recurrence coefficients for the Laguerre polynomials
def recurrence(N,shift=0.,scale=1.):

    from numpy import arange
    return recurrence_ns(arange(N),shift,scale)

# Evaluates the monic Laguerre polynomials at the locations x
def lpoly(x,n,d=0,shift=0.,scale=1.):

    N = _np.max(n)
    [a,b] = recurrence(N+1,shift=shift,scale=scale)
    return opoly1.eval_opoly(x,n,a,b,d)

# Evaluates the L2 normalized Laguerre polynomials at the locations x
def lpolyn(x,n,d=0,shift=0.,scale=1.):

    N = _np.max(n)
    [a,b] = recurrence(N+2,shift=shift,scale=scale)
    return opoly1.eval_opolyn(x,n,a,b,d)

# Returns the N-point Laguerre-Gauss quadrature rule
def gquad(N,shift=0.,scale=1.):

    [a_s,b_s] = recurrence(N,shift=shift,scale=scale)
    return opoly1.opoly_gq(a_s,b_s,N)

# Returns the N-point Laguerre-Gauss-Radau quadrature rule over the interval
# (-scale,scale)+shift
def grquad(N,r0=0.,shift=0,scale=1) : 

    [a_s,b_s] = recurrence(N,shift=shift,scale=scale)
    return opoly1.opoly_grq(a_s,b_s,N,r0=r0)

# Returns the weight function for the Laguerre polynomials evaluated at a
# particular location
def weight(x,shift=0.,scale=1.):
    from numpy import exp
    xt = (x-shift)/scale
    return exp(-xt)

# Returns the square root of the weight function for
# the Laguerre polynomials evaluated at a particular location
def w_sqrt(x,shift=0.,scale=1.):
    from numpy import exp
    xt = (x-shift)/scale
    return exp(-xt/2.)

# Returns the derivative of the square root of the weight function for
# the Laguerre polynomials evaluated at a particular location
def dw_sqrt(x,mu=0.,shift=0.,scale=1.):
    from numpy import exp
    xt = (x-shift)/scale
    return -1/2.*exp(-xt/2.)/scale

########################################################
#                 HELPER FUNCTIONS                     #
########################################################
