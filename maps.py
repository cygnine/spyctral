# Module containing the maps to/from x-space that are used for the generalized
# Wiener rational functions

import numpy as _np

__all__ = ['x2theta']

# Defines the x-theta mapping: takes x -> theta
def x2theta(x,shift=0.,scale=1.):
    
    y = (x-shift)/scale
    theta = -(y-1j)/(y+1j)
    return _np.log(theta).imag

# Defines the theta-x mapping: takes theta -> x
def theta2x(theta,shift=0.,scale=1.):

    return _np.tan(theta/2)*scale + shift

# Defines the theta-x jacobian
def dthetadx(x,shift=0.,scale=1.):

    y = (x-shift)/scale
    return 2./(y**2+1.)/scale

# Defines the x-r mapping: takes x -> r
def x2r(x,shift=0.,scale=1.):

    y = (x-shift)/scale
    return (1-x**2)/(1+x**2)

# Defines the x-r mapping: takes r -> x
def r2x(r,shift=0.,scale=1.):

    return _np.sqrt((1-r)/(1+r))*scale+shift
