#! /usr/bin/env python
# 
# Functions for evaluating coefficients regarding Laguerre polynomials

__all__ = ['recurrence_range',]

def recurrence_range(N,alpha=0.):
# Returns the first N Laguerre recurrence coefficients

    return recurrence(range(N),alpha=alpha)

def recurrence(ns,alpha=0):
# Returns recurrence coefficients for certain values of n
    from numpy import array, zeros, ones, any, arange
    from scipy.special import gamma

    if type(ns) != list:
        ns = [ns]

    ns = array(ns)
    alpha = float(alpha)
    a_s = zeros(ns.shape)
    b_s = zeros(ns.shape)

    b_s[ns==0] = gamma(1+alpha)

    a_s = 2*ns + alpha + 1

    flags = ns!=0
    n = ns[flags]
    b_s[flags] = n*(n+alpha)

    return [a_s,b_s]
