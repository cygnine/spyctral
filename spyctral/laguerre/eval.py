#!/usr/bin/env python
""" 
Package for evaluation of Laguerre polynomials
"""

def laguerre_polynomial(x,n,alpha=0.,d=0, normalization='normal',scale=1., shift=0.) :
    """
    Evaluates the Laguerre polynomials of class alpha, order n (list)
    at the points x (list/array). The normalization is specified by the keyword
    argument. Possibilities:
        'normal', 'normalized':
        'monic':
    """
    from numpy import array, ndarray, ones, sqrt
    from spyctral.laguerre.coeffs import recurrence_range
    from spyctral.opoly1d.eval import eval_opoly, eval_normalized_opoly
    from spyctral.common.maps import physical_scaleshift as pss
    from spyctral.common.maps import standard_scaleshift as sss

    x = array(x)
    if all([type(n) != dtype for dtype in [list,ndarray]]):
        # Then it's probably an int
        n = array([n])
    n = array(n)

    [a,b] = recurrence_range(max(n)+2,alpha=alpha)

    # dummy function for scaling
    def scale_ones(n,alpha):
        return ones([1,len(n)])

    # Set up different normalizations
    normal_fun = {'normal': eval_normalized_opoly,
                  'normalized': eval_normalized_opoly,
                  'monic': eval_opoly}

    normal_scale = {'normal': scale_ones,
                    'normalized': scale_ones,
                    'monic': scale_ones}

    # Shift to standard interval and use opoly 3-term recurrence
    normalization = normalization.lower()
    sss(x,scale=scale,shift=shift)
    temp = normal_fun[normalization](x,n,a,b,d=d)
    temp *= normal_scale[normalization](n,alpha)
    pss(x,scale=scale,shift=shift)

    return temp

def laguerre_function(x,n,alpha=0.,scale=1.,shift=0.):
    """
    Evaluates the L^2-orthonormalized Laguerre functions: i.e. the Laguerre
    polynomials with the weight function multiplicatively distributed to the
    polynomials.
    """
    from spyctral.laguerre.weight import sqrt_weight

    lpolys = laguerre_polynomial(x=x,n=n,alpha=alpha,scale=scale,shift=shift)
    w = sqrt_weight(x=x,alpha=alpha,shift=shift,scale=scale)

    return (lpolys.T*w).T

def laguerre_polynomial_derivative(x,n,alpha=0.,normalization='normal',scale=1.,shift=0.):
    """
    Evaluates the derivative of the Laguerre polynomial with specified
    normalization. Although this is possible using laguerre_polynomial, this
    function evaluates the polynomial of order (alpha+1) and scales it
    accordingly. 
    """

    print "not implemented yet"
    return None

def laguerre_function_derivative(x,n,alpha=0.,scale=1.,shift=0.):
    """
    Evaluates the derivative of the L^2-orthonormalized Laguerre functions: i.e.
    the Laguerre polynomials with the weight function multiplicatively
    distributed to the polynomials.
    """
    from spyctral.laguerre.weight import sqrt_weight, dsqrt_weight

    w = sqrt_weight(x=x,alpha=alpha,shift=shift,scale=scale)
    dw = dsqrt_weight(x=x,alpha=alpha,shift=shift,scale=scale)

    ps = laguerre_polynomial(x=x,n=n,alpha=alpha,scale=scale,shift=shift)
    dps = laguerre_polynomial(x=x,n=n,d=1,alpha=alpha,scale=scale,shift=shift)

    # Yay product rule
    return (w*dps.T + dw*ps.T).T

############### ALIASES #################
lpoly = laguerre_polynomial
lfun = laguerre_function
dlpoly = laguerre_polynomial_derivative
dlfun = laguerre_function_derivative
