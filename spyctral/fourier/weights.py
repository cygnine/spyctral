#!/usr/bin/env python
"""
* File Name : weights.py

* Creation Date : 2009-06-17

* Created By : Akil Narayan

* Last Modified : Wed 17 Jun 2009 05:03:53 PM EDT

* Purpose : Contains information about weight functions
"""

# Defines the Szego-Fourier weight
def weight(theta,gamma=0.,delta=0.,shift=0.,scale=1.):
    from numpy import array, cos
    from spyctral.common.maps import standard_scaleshift as sss
    from spyctral.common.maps import physical_scaleshift as pss

    theta = array(theta)
    sss(theta,scale=scale,shift=shift)
    w = ((1-cos(theta))**delta)*((1+cos(theta))**gamma)
    pss(theta,scale=scale,shift=shift)
    return w

# Defines the conjugate-biased square-root weight function for the Szego-Fourier basis sets
def sqrt_weight_bias(theta,gamma=0.,delta=0.,shift=0.,scale=1.):
    from numpy import cos, sin, exp
    from spyctral.common.maps import standard_scaleshift as sss
    from spyctral.common.maps import physical_scaleshift as pss
    from scipy import pi
    from scipy import power as pw

    sss(theta,scale=scale,shift=shift)
    phase = exp(1j*(gamma+delta)/2.*(pi-theta))
    w = phase*( pw(sin(theta/2.),delta) * \
                pw(cos(theta/2.),gamma)) *\
              2**((gamma+delta)/2.)
    pss(theta,scale=scale,shift=shift)

    return w

# Defines the derivative of the wtheta_sqrt function
# NOTE: the bias is weird for delta = 0: we take abs(sin(theta/2.)) instead of
# sin(theta/2.)
def dsqrt_weight_bias(theta,gamma=0.,delta=0.,shift=0.,scale=1.):
    from numpy import cos, sin, exp, abs
    from spyctral.common.maps import standard_scaleshift as sss
    from spyctral.common.maps import physical_scaleshift as pss
    from scipy import pi

    sss(theta,scale=scale,shift=shift)
    phase = exp(1j*(gamma+delta)/2.*(pi-theta))*\
            2**((gamma+delta-4)/2.)*\
            abs(sin(theta/2.))**(delta-1)*\
            cos(theta/2.)**(gamma-1)
    w = phase*( delta*(1+exp(-1j*theta)) -\
                gamma*(1+exp(1j*theta)))
    pss(theta,scale=scale,shift=shift)

    return w/scale
