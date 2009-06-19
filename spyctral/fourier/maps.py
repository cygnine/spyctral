# Maps module for fourier package
#
# 20090219 -- acn

__all__ = ["r2theta",
           "theta2r"]

def r_to_theta(r):
    from numpy import arccos
    return arccos(r)

def theta_to_r(theta):
    from numpy import cos
    return cos(theta)

def theta_to_rcomplement(theta):
    from numpy import sin
    return sin(theta)
