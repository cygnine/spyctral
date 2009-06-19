# Module containing the maps to/from x-space that are used for the generalized
# Wiener rational functions

# Defines the x-theta mapping: takes x -> theta
def x_to_theta(x,shift=0.,scale=1.):
    from numpy import log
    
    y = (x-shift)/scale
    theta = -(y-1j)/(y+1j)
    return log(theta).imag

# Defines the theta-x mapping: takes theta -> x
def theta_to_x(theta,shift=0.,scale=1.):
    from numpy import tan

    return tan(theta/2)*scale + shift

# Defines the theta-x jacobian
def dtheta_dx(x,shift=0.,scale=1.):

    y = (x-shift)/scale
    return 2./(y**2+1.)/scale

# Defines the x-r mapping: takes x -> r
def x_to_r(x,shift=0.,scale=1.):

    y = (x-shift)/scale
    return (1-x**2)/(1+x**2)

# Defines the x-r mapping: takes r -> x
def r_to_x(r,shift=0.,scale=1.):
    from numpy import sqrt

    return sqrt((1-r)/(1+r))*scale+shift
