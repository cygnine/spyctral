# Maps module for fourier package
#
# 20090219 -- acn

__all__ = ["r2theta",
           "theta2r"]

def r2theta(r):
    from numpy import arccos
    return arccos(r)

def theta2r(theta):
    from numpy import cos
    return cos(theta)

def theta2rc(theta):
    from numpy import sin
    return sin(theta)
