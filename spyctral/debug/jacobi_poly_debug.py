#!/usr/bin/env python
"""
Package for debugging Jacobi polynomials
"""

from __future__ import division

def gauss_test(N,alpha,beta,scale=1.,shift=0.,tol=1e-8):

    from spyctral.jacobi.quad import gq
    from numpy import abs, max, all
    flags = list()
    descriptions = list()
    parameters = list()

    jargs = dict(N=N,alpha=alpha,beta=beta,scale=scale,shift=shift)

    [r,w] = gq(N,alpha=alpha,beta=beta,shift=shift,scale=scale)

    flags.append(max(abs(r-shift)/scale)<1+tol)
    descriptions.append("Gauss quadrature points inside interval")
    parameters.append(jargs)

    flags.append(all(w>=-tol))
    descriptions.append("Gauss quadrature weights positive")
    parameters.append(parameters[-1])

    return flags,descriptions,parameters

def gauss_radau_test(N,alpha,beta,r0=-1.,scale=1.,shift=0.,tol=1e-8):

    from spyctral.jacobi.quad import grq
    from numpy import max, abs, all
    flags = list()
    descriptions = list()
    parameters = list()

    [r,w] = grq(N=N,alpha=alpha,beta=beta,r0=r0,shift=shift,scale=scale)

    jargs = dict(N=N,alpha=alpha,beta=beta,scale=scale,shift=shift,r0=r0)

    flags.append(max(abs(r-shift)/scale)<=1+tol)
    descriptions.append("Gauss-Radau nodes inside interval")
    parameters.append(jargs)

    flags.append(any(abs(r-r0)<tol))
    descriptions.append("Gauss-Radau node")
    parameters.append(parameters[-1])

    flags.append(all(w>=-tol))
    descriptions.append("Gauss-Radau quadrature weights positive")
    parameters.append(parameters[-1])

    return flags,descriptions,parameters

def gauss_lobatto_test(N,alpha,beta,scale=1.,shift=0.,tol=1e-8):

    from spyctral.jacobi.quad import glq
    from numpy import max, abs, all
    flags = list()
    descriptions = list()
    parameters = list()

    [r,w] = glq(N=N,alpha=alpha,beta=beta,shift=shift,scale=scale)

    jargs = dict(N=N,alpha=alpha,beta=beta,scale=scale,shift=shift)

    flags.append(max(abs(r-shift)/scale)<=1+tol)
    descriptions.append("Gauss-Lobatto nodes inside interval")
    parameters.append(jargs)

    flags.append((abs(r[0]-shift+scale)<tol) and (abs(r[-1]-shift-scale)<tol))
    descriptions.append("Gauss-Lobatto nodes at -1, +1")
    parameters.append(parameters[-1])

    flags.append(all(w>=-tol))
    descriptions.append("Gauss-Lobatto quadrature weights positive")
    parameters.append(parameters[-1])

    return flags,descriptions,parameters

def quadrature_test(N,alpha,beta):
    from scipy import rand, randn
    flags = list()
    descriptions = list()
    parameters = list()

    container = [flags,descriptions,parameters]

    tol = 1e-8

    def extend_list_pair(f,g):
        f.extend(g)

    def test_wrapper(fun,**kwargs):
        map(extend_list_pair, container, 
                fun(**kwargs))

    jargs = dict(N=N,alpha=alpha,beta=beta)

    test_wrapper(gauss_test, **jargs)
    test_wrapper(gauss_radau_test, r0=-1., **jargs)
    test_wrapper(gauss_radau_test, r0=1., **jargs)
    test_wrapper(gauss_lobatto_test, **jargs)

    shift = randn()
    scale = 2*rand()
    jargs['shift'] = shift
    jargs['scale'] = scale

    test_wrapper(gauss_test, **jargs)
    test_wrapper(gauss_radau_test, r0=-(scale)+shift, **jargs)
    test_wrapper(gauss_radau_test, r0=scale+shift, **jargs)
    test_wrapper(gauss_lobatto_test, **jargs)

    return flags,descriptions,parameters

def driver():
    """ 
    Driver function for testing Jacobi polynomials
    """

    from scipy import rand, randn
    from numpy import ceil

    flags = list()
    descriptions = list()
    parameters = list()

    container = [flags, descriptions, parameters]

    def extend_list_pair(f,g):
        f.extend(g)

    def test_wrapper(newstuff,oldstuff):
        map(extend_list_pair, oldstuff, newstuff)

    """ Chebyshev case """
    N = int(ceil(150*rand()))
    test_wrapper(quadrature_test(N,alpha=-1/2.,beta=-1/2.), container)

    """ Legendre case """
    N = int(ceil(150*rand()))
    test_wrapper(quadrature_test(N, alpha=0., beta=0.), container)

    """ Random case 1 """
    N = int(ceil(150*rand()))
    alpha = -1 + 11*rand()
    beta = -1 + 11*rand()
    test_wrapper(quadrature_test(N, alpha=alpha,beta=beta), container)

    """ Random case 2 """
    N = int(ceil(150*rand()))
    alpha = -1 + 11*rand()
    beta = -1 + 11*rand()
    test_wrapper(quadrature_test(N, alpha=alpha,beta=beta), container)

    return container
