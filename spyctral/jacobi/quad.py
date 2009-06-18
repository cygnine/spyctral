#!/usr/bin/env python
"""
* File Name : quad.py

* Creation Date : 2009-06-17

* Created By : Akil Narayan

* Last Modified : Wed 17 Jun 2009 03:58:33 PM EDT

* Purpose : 
"""

def gq(N,alpha=-1/2.,beta=-1/2.,shift=0.,scale=1.) : 
# Returns the N-point Jacobi-Gauss(alpha,beta) quadrature rule over the interval
# (-scale,scale)+shift
# The quadrature rule is normalized to reflect the real Jacobian
    from coeffs import recurrence_range
    from spyctral.opoly1d.quad import gq as ogq
    from spyctral import chebyshev
    from spyctral.common.maps import physical_scaleshift as pss

    tol = 1e-12;
    if (abs(alpha+1/2.)<tol) & (abs(beta+1/2.)<tol) :
        return chebyshev.quad.gq(N,shift=shift,scale=scale)
    else :
        [a,b] = recurrence_range(N,alpha,beta)
        temp = ogq(a,b)
        temp[1] *= scale  # scale Jacobian
        pss(temp[0],scale=scale,shift=shift)
        return temp

def grq(N,alpha=-1/2.,beta=-1/2.,r0=None,shift=0.,scale=1.) : 
# Returns the N-point Jacobi-Gauss-Radau(alpha,beta) quadrature rule over the interval
# (-scale,scale)+shift
    from numpy import array
    from coeffs import recurrence_range
    from spyctral.opoly1d.quad import grq as ogrq
    from spyctral import chebyshev
    from spyctral.common.maps import physical_scaleshift as pss
    from spyctral.common.maps import standard_scaleshift as sss

    if r0 is None:
        r0 = [-scale]

    r0 = array(r0)

    tol = 1e-12;
    if (abs(alpha+1/2.)<tol) & (abs(beta+1/2.)<tol) & (abs(abs(r0)-1.)<tol) :
        return chebyshev.quad.grq(N,r0=r0,shift=shift,scale=scale)
    else :
        [a,b] = recurrence_range(N,alpha,beta)
        sss(r0,scale=scale,shift=shift)
        temp = ogrq(a,b,r0=r0)
        pss(r0,scale=scale,shift=shift)
        pss(temp[0],scale=scale,shift=shift)
        temp[1] *= scale
        return temp

def glq(N,alpha=-1/2.,beta=-1/2.,r0=None,shift=0.,scale=1.) : 
# Returns the N-point Jacobi-Gauss-Lobatto(alpha,beta) quadrature rule over the interval
# (-scale,scale)+shift

    from coeffs import recurrence_range
    from numpy import array
    from spyctral import chebyshev
    from spyctral.opoly1d.quad import glq as oglq
    from spyctral.common.maps import physical_scaleshift as pss
    from spyctral.common.maps import standard_scaleshift as sss

    if r0 is None:
        r0 = [-scale,scale]

    r0 = array(r0)
    tol = 1e-12;
    if (abs(alpha+1/2.)<tol) & (abs(beta+1/2.)<tol) :
        return chebyshev.quad.glq(N,shift=shift,scale=scale)
    else :
        [a,b] = recurrence_range(N,alpha,beta)
        sss(r0,scale=scale,shift=shift)
        temp = oglq(a,b,r0=r0)
        pss(r0,scale=scale,shift=shift)
        pss(temp[0],scale=scale,shift=shift)
        temp[1] *= scale
        return temp
