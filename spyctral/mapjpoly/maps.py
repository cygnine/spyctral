#!/usr/bin/env python
#
# Maps module for the mapjpoly package
# Includes a linear scaling of the real axis

#__all__ = ['xtor',
#           'rtox',
#           'drdx',
#           'dxdr']

from spyctral.common.maps import standard_scaleshift as sss
from spyctral.common.maps import physical_scaleshift as pss

# Maps the indices (s,t) to (alpha,beta)
def st_to_ab(s,t):
    return [(2*s-3)/2.,(2*t-3)/2.]

# Map x -----> r(x)
def x_to_r(x,scale=1.,shift=0.):
    from numpy import sqrt

    y = x.copy()
    sss(y,scale=scale,shift=shift)
    return y/sqrt(1+y**2)

# Map r -----> x(r)
def r_to_x(r,scale=1.,shift=0.):
    from numpy import sqrt

    temp = r/sqrt(1-r**2)
    pss(temp,scale=scale,shift=shift)
    return temp

# Jacobian dr/dx(x):
def dr_dx(x,scale=1.,shift=0.):

    y = x.copy()
    sss(y,scale=scale,shift=shift)
    return 1/(1+y**2)**(3/2.)/scale

# Square root Jacobian sqrt(dr/dx)(x)
def sqrt_dr_dx(x,scale=1.,shift=0.):
    from numpy import sqrt

    return sqrt(dr_dx(x,scale=scale,shift=shift))

# Jacobian dx/dr(r):
def dx_dr(r,scale=1.,shift=0.):
    
    #return 1/(1-r**2)**(3/2.)*scale+shift
    return 1/(1-r**2)**(3/2.)*scale

# Returns the Jacobi polynomial weight function of class (alpha,beta)
# evaluated as a function of x
# wjacobi
def jacobi_weight(x,s=1.,t=1.,scale=1.,shift=0.):
    [alpha,beta] = st_to_ab(s,t)
    r = x_to_r(x,scale=scale,shift=shift)
    return ((1-r)**alpha) * ((1+r)**beta)

# Returns the square root of the Jacobi polynomials weight function of
# class (s,t) as a function of x
# sqrt_wjacobi
def sqrt_jacobi_weight(x,s=1.,t=1.,scale=1.,shift=0.):

    [alpha,beta] = st_to_ab(s,t)
    r = x_to_r(x,scale=scale,shift=shift)
    return ((1-r)**(alpha/2.)) * ((1+r)**(beta/2.))

# Returns the weighed Jacobi polynomial weight function of class
# (alpha,beta) used to form the weight function against which the mapped
# Jacobi polynomials are orthogonal
# wjacobiw
def weight(x,s=1.,t=1.,scale=1.,shift=0.):

    return jacobi_weight(x,s=s,t=t,scale=scale,shift=shift)*dr_dx(x,scale=scale,shift=shift)

# Returns the square root of the weighted Jacobi polynomial weight
# function of class (alpha,beta). The primary use of this is in the
# formation of the weighted mapped Jacobi polynomials, which are
# orthogonal under unit weight
# sqrt_wjacobiw
def sqrt_weight(x,s=1.,t=1.,scale=1.,shift=0.):

    return sqrt_jacobi_weight(x,s=s, t=t,scale=scale,shift=shift)*\
            sqrt_dr_dx(x,scale=scale,shift=shift)

# Returns the derivative of the square root weight function: used in
# construction of the derivative of the weighted functions
# dsqrt_wjacobiw
def dsqrt_weight(x,s=1.,t=1.,scale=1.,shift=0.):

    from numpy import sqrt

    [a,b] = st_to_ab(s,t)
    r = x_to_r(x,scale=scale,shift=shift)

    factor = -(a/2.+3/4.)*(1+r) + (b/2.+3/4.)*(1-r)
    # Jacobian for scaling:
    factor /= scale
    #xt = (x-shift)/scale
    xt = x.copy()
    sss(xt,scale=scale,shift=shift)

    return factor/sqrt(1+xt**2)*sqrt_weight(x,s=s,t=t,scale=scale,shift=shift)
