#!/usr/bin/env python
"""
The weight module for spyctral's Laguerre package.
"""

from __future__ import division

def weight(x,alpha=0.,shift=0.,scale=1.):
    """
    Returns the weight function for Laguerre orthogonal polynomials with
    parameter specifications alpha, shift, and scale.
    """
    from spyctral.common.maps import standard_scaleshift as sss
    from spyctral.common.maps import physical_scaleshift as pss
    from numpy import exp
    sss(x,shift=shift,scale=scale)
    wfun = exp(-x)*x**alpha
    wfun *= scale
    pss(x,shift=shift,scale=scale)

    return wfun

def sqrt_weight(x,alpha=0.,shift=0.,scale=1.):
    """
    Returns the square root of the weight function for Laguerre orthogonal
    polynomials with parameter specifications alpha, shift, and scale.
    """
    from spyctral.common.maps import standard_scaleshift as sss
    from spyctral.common.maps import physical_scaleshift as pss
    from numpy import exp, sqrt
    sss(x,shift=shift,scale=scale)
    wfun = exp(-x/2)*x**(alpha/2)
    wfun *= sqrt(scale)
    pss(x,shift=shift,scale=scale)

    return wfun

def dsqrt_weight(x,alpha=0., shift=0., scale=1.):
    """
    Returns the derivative of the square root of the weight function for
    Laguerre orthogonal polynomials with parameter specifications alpha, shift,
    and scale.
    """
    from spyctral.common.maps import standard_scaleshift as sss
    from spyctral.common.maps import physical_scaleshift as pss
    from numpy import exp, sqrt
    sss(x,shift=shift,scale=scale)
    if abs(alpha)<1e-8:
        dwfun = -1/2*exp(-x/2)
    else:
        dwfun = (alpha/2)*x**(alpha/2-1)*exp(-x/2) + \
                x**(alpha/2)*-1/2*exp(-x/2)

    dwfun *= sqrt(scale)
    dwfun /= scale   # Jacobian factor
    pss(x,shift=shift,scale=scale)

    return dwfun
