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
    from numpy import array

    dws = -(s+t)*y**t
    dws = array(dws,dtype=complex)
    if t != 0:
        dws += t*y**(t-1)
    dws /= (y-1j)**(s+t+1.)
    dws *= 2**((s+t)/2.)
    #dws = t*(y-1j) - y*(s+t)
    #if t != 0.:
    #    dws *= y**(t-1.)
    #if (s+t+1.) != 0:
    #    dws *= 1./((y-1j)**(s+t+1.))
    return dws/scale
