#!/usr/bin/env python
""" 
Package for evaluation of Jacobi polynomials
"""

def jacobi_polynomial(x,n,alpha=-1/2.,beta=-1/2.,d=0, normalization='normal',scale=1., shift=0.) :
    """
    Evaluates the Jacobi polynomials of class (alpha,beta), order n (list)
    at the points x (list/array). The normalization is specified by the keyword
    argument. Possibilities:
        'normal', 'normalized':
        'monic':
    """
    from numpy import array, ndarray, ones, sqrt
    from coeffs import recurrence_range
    from spyctral.opoly1d.eval import eval_opoly, eval_normalized_opoly
    from spyctral.common.maps import physical_scaleshift as pss
    from spyctral.common.maps import standard_scaleshift as sss

    x = array(x)
    if all([type(n) != dtype for dtype in [list,ndarray]]):
        # Then it's probably an int
        n = array([n])
    n = array(n)

    [a,b] = recurrence_range(max(n)+2,alpha=alpha,beta=beta)

    # dummy function for scaling
    def scale_ones(n,alpha,beta):
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
    temp *= normal_scale[normalization](n,alpha,beta)
    pss(x,scale=scale,shift=shift)

    return temp

def jacobi_polynomial_derivative(x,n,alpha=-1/2.,beta=-1/2.,normalization='normal',scale=1.,shift=0.):
    """
    Evaluates the derivative of the Jacobi polynomial with specified
    normalization. Although this is possible using jacobi_polynomial, this
    function evaluates the polynomial of order (alpha+1,beta+1) and scales it
    accordingly. 
    """
    from coeffs import zetan
    from numpy import array
    zetas = zetan(array(n),alpha=alpha,beta=beta,normalization=normalization)/scale
    return jacobi_polynomial(x,array(n)-1,alpha+1.,beta+1.,normalization=normalization,\
                 shift=shift,scale=scale)*zetas

############### ALIASES #################
jpoly = jacobi_polynomial
djpoly = jacobi_polynomial_derivative
