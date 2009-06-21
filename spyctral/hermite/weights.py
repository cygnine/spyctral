# Returns the weight function for the Hermite polynomials evaluated at a
# particular location
def weight(x,mu=0.,shift=0.,scale=1.):
    from numpy import abs, exp
    from spyctral.common.maps import physical_scaleshift as pss
    from spyctral.common.maps import standard_scaleshift as sss
    sss(x,shift=shift,scale=scale)
    #xt = (x-shift)/scale
    weight = abs(x)**(2*mu)*exp(-x**2)
    pss(x,shift=shift,scale=scale)
    return  weight

# Returns the square root of the weight function for
# the Hermite polynomials evaluated at a particular location
def sqrt_weight(x,mu=0.,shift=0.,scale=1.):
    from spyctral.common.maps import physical_scaleshift as pss
    from spyctral.common.maps import standard_scaleshift as sss
    from numpy import exp, abs

    sss(x,shift=shift,scale=scale)
    #xt = (x-shift)/scale
    w = exp(-x**2/2)*abs(x)**mu
    pss(x,shift=shift,scale=scale)
    return w

# Returns the derivative of the square root of the weight function for
# the Hermite polynomials evaluated at a particular location
def dsqrt_weight(x,mu=0.,shift=0.,scale=1.):
    from spyctral.common.maps import physical_scaleshift as pss
    from spyctral.common.maps import standard_scaleshift as sss
    from numpy import exp, abs

    sss(x,shift=shift,scale=scale)
    #xt = (x-shift)/scale
    w = exp(-x**2/2)*(mu*abs(x)**(mu-1) - x*abs(x)**mu)
    pss(x,shift=shift,scale=scale)
    return w/scale
