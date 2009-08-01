#!/usr/bin/env python
#
# Jacobi polynomial quadrature package

def gauss_quadrature(N,alpha=-1/2.,beta=-1/2.,shift=0.,scale=1.) : 
# Returns the N-point Jacobi-Gauss(alpha,beta) quadrature rule over the interval
# (-scale,scale)+shift.
# The quadrature rule is *not* normalized in the sense that the affine parameters
# shift and scale are built into the new weight function for which this
# quadrature rule is valid.

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
        pss(temp[0],scale=scale,shift=shift)
        return temp

def gauss_radau_quadrature(N,alpha=-1/2.,beta=-1/2.,r0=None,shift=0.,scale=1.) : 
# Returns the N-point Jacobi-Gauss-Radau(alpha,beta) quadrature rule over the interval
# (-scale,scale)+shift
# For nontrivial values of shift, scale, the weight function associated with
# this quadrature rule is the directly-mapped Jacobi weight + the Jacobian
# factor introduced by scale.

    from numpy import array
    from coeffs import recurrence_range
    from spyctral.opoly1d.quad import grq as ogrq
    from spyctral import chebyshev
    from spyctral.common.maps import physical_scaleshift as pss
    from spyctral.common.maps import standard_scaleshift as sss

    if r0 is None:
        r0 = [-scale+shift]

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
        return temp

def gauss_lobatto_quadrature(N,alpha=-1/2.,beta=-1/2.,r0=None,shift=0.,scale=1.) : 
# Returns the N-point Jacobi-Gauss-Lobatto(alpha,beta) quadrature rule over the interval
# (-scale,scale)+shift
# For nontrivial values of shift, scale, the weight function associated with
# this quadrature rule is the directly-mapped Jacobi weight + the Jacobian
# factor introduced by scale.

    from coeffs import recurrence_range
    from numpy import array
    from spyctral import chebyshev
    from spyctral.opoly1d.quad import glq as oglq
    from spyctral.common.maps import physical_scaleshift as pss
    from spyctral.common.maps import standard_scaleshift as sss

    if r0 is None:
        r0 = array([-scale,scale])+shift

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
        #temp[1] *= scale
        return temp

############### ALIASES #################
gq = gauss_quadrature
grq = gauss_radau_quadrature
glq = gauss_lobatto_quadrature
