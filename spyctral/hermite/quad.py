# Module for quadrature of the Hermite functions
#

#__all__ = []

# Returns the N-point Hermite-Gauss quadrature rule
def gq(N,mu=0.,shift=0.,scale=1.):
    from spyctral.hermite.coeffs import recurrence_range
    from spyctral.opoly1d.quad import gq as ogq
    from spyctral.common.maps import physical_scaleshift as pss

    [a_s,b_s] = recurrence_range(N,mu)
    [x,w] = ogq(a_s,b_s)
    pss(x,scale=scale,shift=shift)
    w *= scale
    return [x,w]

# Returns the N-point Jacobi-Gauss-Radau(mu) quadrature rule over the interval
# (-scale,scale)+shift
def grq(N,mu=0.,r0=0.,shift=0,scale=1) : 
    from spyctral.hermite.coeffs import recurrence_range
    from spyctral.opoly1d.quad import grq as ogrq
    from spyctral.common.maps import physical_scaleshift as pss

    [a_s,b_s] = recurrence_range(N,mu)
    [x,w] = ogrq(a_s,b_s,N)
    pss(x,shift=shift,scale=scale)
    w *= scale
    return [x,w]

# Returns the N-point pi-Gauss quadrature
def pgq(N,mu=0.,scale=1.,shift=0.):

    from numpy import array, zeros, exp
    from spyctral.common.maps import physical_scaleshift as pss
    from spyctral.common.maps import standard_scaleshift as sss

    mu = float(mu) # ???

    [x,w] = gq(N,mu,scale=scale,shift=shift)

    sss(x,shift=shift,scale=scale)
    #xt = (x-shift)/scale
    w /= x**(2*mu)*exp(-x**2)
    pss(x,shift=shift,scale=scale)

    # I DON'T UNDERSTAND THIS
    w *= scale**2

    return [x,w]

"""
# Returns the N-point pi-Gauss-Radau quadrature
# WHAT TO DO WITH DEFAULT QUADRATURE IF R0=0 AND MU!=0 ???
def pgrq(N,r0=0.,mu=0.,scale=1.,shift=0.):

    from opoly1.hermite import grquad
    from numpy import array, zeros, exp

    mu = float(mu)

    [x,w] = grquad(N,r0=r0,mu=mu,scale=scale,shift=shift)
    xt = (x-shift)/scale
    w /= xt**(mu)*exp(-xt**2)

    # I DON'T UNDERSTAND THIS
    w *= scale**2

    return [x,w]
"""
