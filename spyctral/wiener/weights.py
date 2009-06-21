# Evaluates the weight function
def weight(x,s=1.,t=0.,shift=0.,scale=1.):

    from spyctral.fourier.weights import weight as wtheta
    from maps import x_to_theta as x2theta
    from spyctral.common.maps import standard_scaleshift as sss
    from spyctral.common.maps import physical_scaleshift as pss

    theta = x2theta(x,shift=shift,scale=scale)

    y = (x-shift)/scale
    return 2*wtheta(theta,s-1,t)/(1+y**2)

# Evaluates the phase-shifted square root of the weight function
def sqrt_weight_bias(x,s=1.,t=0.,shift=0.,scale=1.):

    y = (x-shift)/scale
    weightsqrt = y**t/(y-1j)**(s+t)
    weightsqrt *= 2**((s+t)/2.)
    return weightsqrt

# Evaluates the derivative of the phase-shifted square root of the weight
# function
def dsqrt_weight_bias(x,s=1.,t=0.,shift=0.,scale=1.):

    y = (x-shift)/scale
    dws = t*(y-1j) - y*(s+t)
    dws *= (y**(t-1))/((y-1j)**(s+t+1.))
    dws *= 2**((s+t)/2.)
    return dws/scale
