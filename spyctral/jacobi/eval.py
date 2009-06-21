#!/usr/bin/env python
"""
* File Name : eval.py

* Creation Date : 2009-06-17

* Created By : Akil Narayan

* Last Modified : Wed 17 Jun 2009 03:41:32 PM EDT

* Purpose :
"""


def jpoly(x,n,alpha=-1/2.,beta=-1/2.,d=0, normalization='normal',scale=1., shift=0.) :
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
    if all([type(n) != test for test in [list,ndarray]]):
        # Then it's probably an int
        n = array([n])
    n = array(n)

    [a,b] = recurrence_range(max(n)+2,alpha=alpha,beta=beta)

    # dummy function for scaling
    def scale_ones(n,alpha,beta):
        return ones(n.shape)

    # Set up different normalizations
    normal_fun = {'normal': eval_normalized_opoly,
                  'normalized': eval_normalized_opoly,
                  'monic': eval_opoly}

    normal_scale = {'normal': lambda n,alpha,beta: 1/sqrt(scale),
                    'normalized': lambda n, alpha,beta: 1/sqrt(scale),
                    'monic': scale_ones}

    # Shift to standard interval and use opoly 3-term recurrence
    normalization = normalization.lower()
    sss(x,scale=scale,shift=shift)
    temp = normal_fun[normalization](x,n,a,b,d=d)
    temp *= normal_scale[normalization](n,alpha,beta)
    pss(x,scale=scale,shift=shift)

    return temp

# Temporary function to evaluate Jacobi derivatives
def djpoly(x,n,alpha=-1/2.,beta=-1/2.,normalization='normal',scale=1.,shift=0.):
    from coeffs import zetan
    from numpy import array
    zetas = zetan(array(n),alpha=alpha,beta=beta,normalization=normalization)/scale
    return jpoly(x,array(n)-1,alpha+1.,beta+1.,normalization=normalization,\
                 shift=shift,scale=scale)*zetas
