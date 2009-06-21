# Module for evaluation of the Hermite functions

#__all__ = []

# Evaluates the monic Hermite polynomials of order mu at the locations x
def hermite_polynomial(x,n,mu=0.,d=0,shift=0.,scale=1.):
    from numpy import max
    from spyctral.hermite.coeffs import recurrence_range
    from spyctral.opoly1d.eval import eval_normalized_opoly
    from spyctral.common.maps import physical_scaleshift as pss
    from spyctral.common.maps import standard_scaleshift as sss

    N = max(n)
    [a,b] = recurrence_range(N+1,mu)
    sss(x,shift=shift,scale=scale)
    temp = eval_normalized_opoly(x,n,a,b,d)
    pss(x,shift=shift,scale=scale)
    return temp

# Evaluates the L2 normalized Hermite functions
def hermite_function(x,ns,mu=0.,scale=1.,shift=0.):

    from numpy import array, zeros, exp, isnan, Inf
    from spyctral.common.maps import physical_scaleshift as pss
    from spyctral.common.maps import standard_scaleshift as sss

    ns = array(ns)
    N = ns.size
    mu = float(mu)

    ps = hermite_polynomial(x,ns,mu,scale=scale,shift=shift)
    sss(x,shift=shift,scale=scale)
    #x = (x-shift)/scale
    w = x**(mu/4)*exp(-1/2.*x**2)
    pss(x,shift=shift,scale=scale)
    # I DON'T UNDERSTAND THIS
    ps /= scale
    # WARNING: THIS CAUSES PROBLEMS FOR VERY VERY LARGE N: it eliminates
    # nan's
    wflags = (w==0)
    ps[wflags,:] = 0.
    return (ps.T*w).T

# Evaluates the derivative of the L2 normalized Hermite functions
def dhermite_function(x,ns,mu=0.,scale=1.,shift=0.):

    from spyctral.hermite.weights import dsqrt_weight
    from numpy import array, zeros, exp,sqrt
    from spyctral.common.maps import physical_scaleshift as pss
    from spyctral.common.maps import standard_scaleshift as sss

    ns = array(ns)
    N = ns.size
    mu = float(mu)

    ps = hermite_polynomial(x,ns,mu,scale=scale,shift=shift)
    dps = hermite_polynomial(x,ns,mu,d=1,scale=scale,shift=shift)
    sss(x,shift=shift,scale=scale)
    #x = (x-shift)/scale
    w = x**(mu)*exp(-1/2.*x**2)
    pss(x,shift=shift,scale=scale)
    dw = dsqrt_weight(x,mu=mu,shift=shift,scale=scale)

    # yay product rule
    ps = (dps.T*w).T + (ps.T*dw).T

    # I DON'T UNDERSTAND THIS (part of hfunn)
    ps /= scale
    # WARNING: THIS CAUSES PROBLEMS FOR VERY VERY LARGE N: it eliminates
    # nan's
    wflags = (w==0)
    ps[wflags,:] = 0.
    return ps
