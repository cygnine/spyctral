#!/usr/bin/env python
"""
Laguerre polynomial quadrature package.
"""

from __future__ import division

def gauss_quadrature(N,alpha=0.,shift=0.,scale=1.) : 
    """
    Returns the N-point Laguerre-Gauss(alpha) quadrature rule over the interval
    (shift,inf).  The quadrature rule is *not* normalized in the sense
    that the affine parameters shift and scale are built into the new weight
    function for which this quadrature rule is valid.
    """

    from spyctral.laguerre.coeffs import recurrence_range
    from spyctral.opoly1d.quad import gq as ogq
    from spyctral.common.maps import physical_scaleshift as pss

    [a,b] = recurrence_range(N,alpha)
    temp = ogq(a,b)
    pss(temp[0],scale=scale,shift=shift)
    return temp

def gauss_radau_quadrature(N,alpha=0.,r0=None,shift=0.,scale=1.): 
    """
    Returns the N-point Laguerre-Gauss-Radau(alpha) quadrature rule over
    the interval (shift,inf) For nontrivial values of shift, scale, the
    weight function associated with this quadrature rule is the directly-mapped
    Laguerre weight + the Jacobian factor introduced by scale.
    """

    from numpy import array
    from spyctral.laguerre.coeffs import recurrence_range
    from spyctral.opoly1d.quad import grq as ogrq
    from spyctral.common.maps import physical_scaleshift as pss
    from spyctral.common.maps import standard_scaleshift as sss

    if r0 is None:
        r0 = 0+shift

    r0 = array(r0)

    [a,b] = recurrence_range(N,alpha)
    sss(r0,scale=scale,shift=shift)
    temp = ogrq(a,b,r0=r0)
    pss(r0,scale=scale,shift=shift)
    pss(temp[0],scale=scale,shift=shift)
    return temp

def pi_gauss_quadrature(N,alpha=0.,shift=0.,scale=1.): 
    """
    Returns the N-point pi-Laguerre-Gauss(alpha) quadrature rule over the interval
    (shift,\inf). This returns a quadrature rule valid for integration
    on the semi-infinite interval under unit weight measure.
    """
    from spyctral.laguerre.weights import weight

    [x,w] = gauss_quadrature(N=N,alpha=alpha,shift=shift,scale=scale)

    w /= weight(x,alpha=alpha,shift=shift,scale=scale)
    return x,w

def pi_gauss_radau_quadrature(N, alpha=0., r0=None, shift=0., scale=1.):
    """
    Returns the N-point pi-Laguerre-Gauss-Radau(alpha) quadrature rule over the
    interval (shift,\inf). This returns a quadrature rule valid for integration
    on the semi-infinite interval under unit weight measure.
    """
    from spyctral.laguerre.weights import weight

    [x,w] = gauss_radau_quadrature(N=N,alpha=alpha,r0=r0,shift=shift,scale=scale)

    w /= weight(x,alpha=alpha,shift=shift,scale=scale)
    return x,w

############### ALIASES #################
gq = gauss_quadrature
pgq = pi_gauss_quadrature
grq = gauss_radau_quadrature
pgrq = pi_gauss_radau_quadrature
