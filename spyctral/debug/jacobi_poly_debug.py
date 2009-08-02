#!/usr/bin/env python
"""
Package for debugging Jacobi polynomials
"""

from __future__ import division
from spyctral.debug.debug_tools import ValidationTest, ValidationContainer 

def gauss_test(N,alpha,beta,scale=1.,shift=0.,tol=1e-8):

    from spyctral.jacobi.quad import gq
    from numpy import abs, max, all
    tests = ValidationContainer()

    def val1(data,**kwargs):
        r = data[0]
        return max(abs(r-shift)/scale)<1+tol

    jargs = dict(N=N,alpha=alpha,beta=beta,scale=scale,shift=shift)

    tests.append(ValidationTest(\
         description="Gauss quadrature points inside interval",
         parameters = jargs,
         data_generator = gq,
         validator = val1))
    
    def val2(data,**kwargs):
        w = data[1]
        return all(w>=-tol)

    tests.append(ValidationTest(\
            description = "Gauss quadrature weights positive",
            parameters = jargs,
            data_generator = gq,
            validator = val2))
    
    return tests

def gauss_radau_test(N,alpha,beta,r0=-1.,scale=1.,shift=0.,tol=1e-8):

    from spyctral.jacobi.quad import grq
    from numpy import max, abs, all

    tests = ValidationContainer()

    jargs = dict(N=N,alpha=alpha,beta=beta,scale=scale,shift=shift,r0=r0)

    def val1(data,**kwargs):
        r = data[0]
        return max(abs(r-shift)/scale)<=1+tol

    tests.append(ValidationTest(\
         description="Gauss-Radau nodes inside interval",
         parameters = jargs,
         data_generator = grq,
         validator = val1))

    def val2(data,**kwargs):
        r = data[0]
        return any(abs(r-r0)<tol)

    tests.append(ValidationTest(\
         description="Gauss-Radau node",
         parameters = jargs,
         data_generator = grq,
         validator = val2))

    def val3(data,**kwargs):
        w = data[1]
        return all(w>=-tol)

    tests.append(ValidationTest(\
         description="Gauss-Radau weights positive",
         parameters = jargs,
         data_generator = grq,
         validator = val3))

    return tests

def gauss_lobatto_test(N,alpha,beta,scale=1.,shift=0.,tol=1e-8):

    from spyctral.jacobi.quad import glq
    from numpy import max, abs, all
    tests = ValidationContainer()

    jargs = dict(N=N,alpha=alpha,beta=beta,scale=scale,shift=shift)

    def val1(data,**kwargs):
        r = data[0]
        return max(abs(r-shift)/scale)<=1+tol

    tests.append(ValidationTest(\
         description="Gauss-Lobatto nodes inside interval",
         parameters = jargs,
         data_generator = glq,
         validator = val1))

    def val2(data,**kwargs):
        r = data[0]
        return (abs(r[0]-shift+scale)<tol) and (abs(r[-1]-shift-scale)<tol)

    tests.append(ValidationTest(\
         description="Gauss-Lobatto nodes at -1, +1",
         parameters = jargs,
         data_generator = glq,
         validator = val2))

    def val3(data,**kwargs):
        w = data[1]
        return all(w>=-tol)

    tests.append(ValidationTest(\
         description="Gauss-Lobatto quadrature weights positive",
         parameters = jargs,
         data_generator = glq,
         validator = val3))

    return tests

def quadrature_test(N,alpha,beta):
    from scipy import rand, randn

    tol = 1e-8
    quad_tests = ValidationContainer()

    jargs = dict(N=N,alpha=alpha,beta=beta)

    quad_tests.extend(gauss_test(tol=tol,**jargs))
    quad_tests.extend(gauss_radau_test(r0=-1.,tol=tol,**jargs))
    quad_tests.extend(gauss_radau_test(r0=1.,tol=tol,**jargs))
    quad_tests.extend(gauss_lobatto_test(tol=tol,**jargs))

    shift = randn()
    scale = 2*rand()
    jargs['shift'] = shift
    jargs['scale'] = scale

    quad_tests.extend(gauss_test(tol=tol,**jargs))
    quad_tests.extend(gauss_radau_test(r0=-(scale)+shift, tol=tol,**jargs))
    quad_tests.extend(gauss_radau_test(r0=scale+shift, tol=tol,**jargs))
    quad_tests.extend(gauss_lobatto_test(tol=tol,**jargs))

    return quad_tests

def driver():
    """ 
    Driver function for testing Jacobi polynomials
    """

    from scipy import rand, randn
    from numpy import ceil

    all_tests = ValidationContainer()

    """ Chebyshev case """
    N = int(ceil(150*rand()))
    all_tests.extend(quadrature_test(N,alpha=-1/2., beta=-1/2.))

    """ Legendre case """
    N = int(ceil(150*rand()))
    all_tests.extend(quadrature_test(N, alpha=0., beta=0.))

    """ Random case 1 """
    N = int(ceil(150*rand()))
    alpha = -1 + 11*rand()
    beta = -1 + 11*rand()
    all_tests.extend(quadrature_test(N, alpha=alpha, beta=beta))

    """ Random case 2 """
    N = int(ceil(150*rand()))
    alpha = -1 + 11*rand()
    beta = -1 + 11*rand()
    all_tests.extend(quadrature_test(N, alpha=alpha, beta=beta))

    return all_tests
