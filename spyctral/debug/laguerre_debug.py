#!/usr/bin/env python
"""
Package for debugging Laguerre polynomials
"""

from __future__ import division
from spyctral.debug.debug_tools import ValidationTest, ValidationContainer 

def gauss_test(N,alpha,scale=1.,shift=0.,tol=1e-8):
    """
    Defines all the tests for Gauss quadrature nodes. Returns a
    ValidationContainer.
    """

    from spyctral.laguerre.quad import gq
    from numpy import abs, max, all
    tests = ValidationContainer()

    def val1(data,**kwargs):
        from spyctral.laguerre.misc import interval
        r = data[0]
        lint = interval(scale=scale,shift=shift)
        return all((r-lint[0])>-tol)

    jargs = dict(N=N,alpha=alpha,scale=scale,shift=shift)

    tests.append(ValidationTest(\
         description="Laguerre Gauss quadrature points inside interval",
         parameters = jargs,
         data_generator = gq,
         validator = val1))
    
    def val2(data,**kwargs):
        w = data[1]
        return all(w>=-tol)

    tests.append(ValidationTest(\
            description = "Laguerre Gauss quadrature weights positive",
            parameters = jargs,
            data_generator = gq,
            validator = val2))

    def dgen1(**kwargs):
        from spyctral.laguerre.eval import laguerre_polynomial
        [x,w] = gq(**kwargs)
        ps = laguerre_polynomial(x,range(2*N),alpha=alpha,scale=scale,shift=shift)
        ps[:,0] = 0
        return [w,ps]

    def val3(data,**kwargs):
        from numpy import dot
        [w,ps] = data
        errs = dot(w.conj().T,ps)
        return max(abs(errs))<tol

    tests.append(ValidationTest(\
            description = "Laguerre Gauss quadrature integration accuracy",
            parameters = jargs,
            data_generator = dgen1,
            validator = val3))

    return tests

def pi_gauss_test(N,alpha,scale=1.,shift=0.,tol=1e-8):
    """
    Defines all the tests for pi-Gauss quadrature nodes. Returns a
    ValidationContainer.
    """

    from spyctral.laguerre.quad import pgq
    from numpy import abs, max, all
    tests = ValidationContainer()

    def val1(data,**kwargs):
        from spyctral.laguerre.misc import interval
        r = data[0]
        lint = interval(scale=scale,shift=shift)
        return all((r-lint[0])>-tol)

    jargs = dict(N=N,alpha=alpha,scale=scale,shift=shift)

    tests.append(ValidationTest(\
         description="Laguerre pi-Gauss quadrature points inside interval",
         parameters = jargs,
         data_generator = pgq,
         validator = val1))
    
    def val2(data,**kwargs):
        w = data[1]
        return all(w>=-tol)

    tests.append(ValidationTest(\
            description = "Laguerre pi-Gauss quadrature weights positive",
            parameters = jargs,
            data_generator = pgq,
            validator = val2))

    def dgen1(**kwargs):
        from spyctral.laguerre.eval import laguerre_function
        from spyctral.laguerre.weights import sqrt_weight
        [x,w] = pgq(**kwargs)
        ps = laguerre_function(x,range(2*N),alpha=alpha,scale=scale,shift=shift)
        ps[:,0] = 0
        ps = (ps.T*sqrt_weight(x,alpha=alpha,scale=scale,shift=shift)).T
        return [w,ps]

    def val3(data,**kwargs):
        from numpy import dot
        [w,ps] = data
        errs = dot(w.conj().T,ps)
        return max(abs(errs))<tol

    tests.append(ValidationTest(\
            description = "Laguerre pi-Gauss quadrature integration accuracy",
            parameters = jargs,
            data_generator = dgen1,
            validator = val3))

    return tests

def gauss_radau_test(N,alpha,r0=0.,scale=1.,shift=0.,tol=1e-8):

    from spyctral.laguerre.quad import grq
    from numpy import max, abs, all

    tests = ValidationContainer()

    jargs = dict(N=N,alpha=alpha,scale=scale,shift=shift,r0=r0)

    def val1(data,**kwargs):
        from spyctral.laguerre.misc import interval
        r = data[0]
        lint = interval(scale=scale,shift=shift)
        return all((r-lint[0])>-tol)

    tests.append(ValidationTest(\
         description="Laguerre Gauss-Radau nodes inside interval",
         parameters = jargs,
         data_generator = grq,
         validator = val1))

    def val2(data,**kwargs):
        r = data[0]
        return any(abs(r-r0)<tol)

    tests.append(ValidationTest(\
         description="Laguerre Gauss-Radau node",
         parameters = jargs,
         data_generator = grq,
         validator = val2))

    def val3(data,**kwargs):
        w = data[1]
        return all(w>=-tol)

    tests.append(ValidationTest(\
         description="Laguerre Gauss-Radau weights positive",
         parameters = jargs,
         data_generator = grq,
         validator = val3))

    def dgen1(**kwargs):
        from spyctral.laguerre.eval import laguerre_polynomial
        from spyctral.laguerre.quad import grq
        [x,w] = grq(**kwargs)
        ps = laguerre_polynomial(x,range(2*N-1),alpha=alpha,scale=scale,shift=shift)
        ps[:,0] = 0
        return [w,ps]

    def val4(data,**kwargs):
        from numpy import dot
        [w,ps] = data
        errs = dot(w.conj().T,ps)
        return max(abs(errs))<tol

    tests.append(ValidationTest(\
            description = "Laguerre Gauss-Radau quadrature integration accuracy",
            parameters = jargs,
            data_generator = dgen1,
            validator = val4))

    return tests

def pi_gauss_radau_test(N,alpha,r0=0.,scale=1.,shift=0.,tol=1e-8):

    from spyctral.laguerre.quad import pgrq
    from numpy import max, abs, all

    tests = ValidationContainer()

    jargs = dict(N=N,alpha=alpha,scale=scale,shift=shift,r0=r0)

    def val1(data,**kwargs):
        from spyctral.laguerre.misc import interval
        r = data[0]
        lint = interval(scale=scale,shift=shift)
        return all((r-lint[0])>-tol)

    tests.append(ValidationTest(\
         description="Laguerre pi-Gauss-Radau nodes inside interval",
         parameters = jargs,
         data_generator = pgrq,
         validator = val1))

    def val2(data,**kwargs):
        r = data[0]
        return any(abs(r-r0)<tol)

    tests.append(ValidationTest(\
         description="Laguerre pi-Gauss-Radau node",
         parameters = jargs,
         data_generator = pgrq,
         validator = val2))

    def val3(data,**kwargs):
        w = data[1]
        return all(w>=-tol)

    tests.append(ValidationTest(\
         description="Laguerre pi-Gauss-Radau weights positive",
         parameters = jargs,
         data_generator = pgrq,
         validator = val3))

    def dgen1(**kwargs):
        from spyctral.laguerre.eval import laguerre_function
        from spyctral.laguerre.weights import sqrt_weight
        from spyctral.laguerre.quad import pgrq
        [x,w] = pgrq(**kwargs)
        ps = laguerre_function(x,range(2*N-1),alpha=alpha,scale=scale,shift=shift)
        ps[:,0] = 0
        ps = (ps.T*sqrt_weight(x,alpha=alpha,scale=scale,shift=shift)).T
        return [w,ps]

    def val4(data,**kwargs):
        from numpy import dot
        [w,ps] = data
        errs = dot(w.conj().T,ps)
        return max(abs(errs))<tol

    tests.append(ValidationTest(\
            description = "Laguerre pi-Gauss-Radau quadrature integration accuracy",
            parameters = jargs,
            data_generator = dgen1,
            validator = val4))

    return tests

def mass_test(N,alpha,scale=1.,shift=0.,tol=1e-8):

    tests = ValidationContainer()
    jargs = dict(N=N,alpha=alpha,scale=scale,shift=shift)

    def dgen1(**kwargs):
        from spyctral.laguerre.quad import gq
        from spyctral.laguerre.eval import laguerre_polynomial

        [r,w] = gq(**kwargs)
        ps = laguerre_polynomial(r,range(N),alpha=alpha,scale=scale,shift=shift)
        return  [w,ps]

    def val1(data,**kwargs):
        from numpy import dot, eye
        from numpy.linalg import norm
        [w,ps] = data
        mass = dot(ps.conj().T*w, ps)
        temp = min([N,10])
        return norm(mass[:temp,:temp] - eye(temp))<tol

    tests.append(ValidationTest(\
            description = "Laguerre Orthonormal poly mass matrix",
            parameters = jargs,
            data_generator = dgen1,
            validator = val1))

    def dgen2(**kwargs):
        from spyctral.laguerre.quad import pgq
        from spyctral.laguerre.eval import laguerre_function

        [r,w] = pgq(**kwargs)
        ps = laguerre_function(r,range(N),alpha=alpha,scale=scale,shift=shift)
        return [w,ps]

    def val2(data,**kwargs):
        from numpy import dot, eye
        from numpy.linalg import norm
        [w,ps] = data
        mass = dot(ps.conj().T*w, ps)
        temp = min([N,50])
        return norm(mass[:temp,:temp] - eye(temp))<tol

    return tests

def evaluation_test(N,alpha):
    from scipy import rand, randn

    tol = 1e-6
    eval_tests = ValidationContainer()

    jargs = dict(N=N,alpha=alpha)

    eval_tests.extend(mass_test(tol=tol,**jargs))

    shift = randn()
    scale = 2*rand()
    jargs['shift'] = shift
    jargs['scale'] = scale

    eval_tests.extend(mass_test(tol=tol,**jargs))

    return eval_tests

def quadrature_test(N,alpha):
    from scipy import rand, randn

    tol = 1e-8
    quad_tests = ValidationContainer()

    jargs = dict(N=N,alpha=alpha)

    quad_tests.extend(gauss_test(tol=tol,**jargs))
    quad_tests.extend(pi_gauss_test(tol=tol,**jargs))
    quad_tests.extend(gauss_radau_test(r0=0.,tol=tol,**jargs))
    if alpha==0:
        quad_tests.extend(pi_gauss_radau_test(r0=0.,tol=tol,**jargs))

    shift = randn()
    scale = 2*rand()
    jargs['shift'] = shift
    jargs['scale'] = scale

    quad_tests.extend(gauss_test(tol=tol,**jargs))
    quad_tests.extend(pi_gauss_test(tol=tol,**jargs))
    quad_tests.extend(gauss_radau_test(r0=shift, tol=tol,**jargs))
    if alpha==0:
        quad_tests.extend(pi_gauss_radau_test(r0=shift, tol=tol,**jargs))

    return quad_tests

def driver():
    """ 
    Driver function for testing Laguerre polynomials
    """

    from scipy import rand, randn
    from numpy import ceil

    all_tests = ValidationContainer()

    """ Laguerre case """
    N = int(ceil(50*rand()))
    all_tests.extend(quadrature_test(N,alpha=0.))
    all_tests.extend(evaluation_test(N,alpha=0.))

    """ Random generalized case 1 """
    N = int(ceil(50*rand()))
    alpha = 10*rand()
    all_tests.extend(quadrature_test(N, alpha=alpha))
    all_tests.extend(evaluation_test(N, alpha=alpha))

    """ Random generalized case 2 """
    N = int(ceil(50*rand()))
    alpha = 10*rand()
    all_tests.extend(quadrature_test(N, alpha=alpha))
    all_tests.extend(evaluation_test(N, alpha=alpha))

    return all_tests
