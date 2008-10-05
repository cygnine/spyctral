#! /usr/bin/env python
# 
# Functions for evaluation and quadrature of Jacobi polynomials

__all__ = ['recurrence','jpoly','jpolyn','gquad']

import numpy as _np
import scipy.special as _nf
#from opoly1 import eval_opoly, eval_opolyn, opoly_gq, opoly_grq, opoly_glq
import opoly1
import cheb1

# Returns the first N Jacobi recurrence coefficients
def recurrence(N,alpha=-1/2.,beta=-1/2.,shift=0,scale=1) :

    alpha = float(alpha)
    beta = float(beta)
    a_s = (beta**2-alpha**2)*_np.ones([N])
    b_s = _np.zeros([N])

    a_s[0] = (beta-alpha)/(alpha+beta+2);
    b_s[0] = 2**(alpha+beta+1)*_nf.gamma(alpha+1)*_nf.gamma(beta+1)/_nf.gamma(alpha+beta+2)
    
    for q in range(2,N+1) :
        k = q-1
        a_s[k] = a_s[k]/((2*k+alpha+beta)*(2*k+alpha+beta+2))

        if k==1 :
            b_s[k] = 4*k*(k+alpha)*(k+beta)/((2*k+alpha+beta)**2*(2*k+alpha+beta+1))
        else :
            num = 4*k*(k+alpha)*(k+beta)*(k+alpha+beta)
            den =  (2*k+alpha+beta)**2*(2*k+alpha+beta+1)*(2*k+alpha+beta-1)
            b_s[k] = num/den

    return [a_s,b_s]
        
    # Still have recurrence_scaleshift to deal with


# Evaluates the monic Jacobi polynomials of class (alpha,beta), order n (list)
# at the points x (list)
def jpoly(x,n,alpha=-1/2.,beta=-1/2.,d=0) :
    N = _np.max(n);
    [a,b] = recurrence(N+1,alpha,beta)
    return opoly1.eval_opoly(x,n,a,b,d)

# Evaluates the L^2-normalized Jacobi polynomials of class (alpha,beta), order n (list)
# at the points x (list)
def jpolyn(x,n,alpha=-1/2.,beta=-1/2.,d=0) :
    N = _np.max(n);
    [a,b] = recurrence(N+2,alpha,beta)
    return opoly1.eval_opolyn(x,n,a,b,d)

# Returns the N-point Jacobi-Gauss(a,b) quadrature rule over the interval
# (-scale,scale)+shift
def gquad(N,a=-1/2.,b=-1/2.,shift=0,scale=1) : 

    tol = 1e-12;
    if (abs(a+1/2.)<tol) & (abs(b+1/2.)<tol) :
        return cheb1.gquad(N,shift,scale)
    else :
        [a_s,b_s] = recurrence(N,a,b,shift,scale)
        return opoly1.opoly_gq(a_s,b_s,N)

# Returns the N-point Jacobi-Gauss-Radau(a,b) quadrature rule over the interval
# (-scale,scale)+shift
def grquad(N,a=-1/2.,b=-1/2.,r0=-1.,shift=0,scale=1) : 

    tol = 1e-12;
    if (abs(a+1/2.)<tol) & (abs(b+1/2.)<tol) & (abs(r0)-1<tol) :
        return cheb1.grquad(N,shift,scale)
    else :
        [a_s,b_s] = recurrence(N,a,b,shift,scale)
        return opoly1.opoly_grq(a_s,b_s,N,r0=r0)

# Returns the N-point Jacobi-Gauss-Lobatto(a,b) quadrature rule over the interval
# (-scale,scale)+shift
def glquad(N,a=-1/2.,b=-1/2.,r0=[-1.,1.],shift=0,scale=1) : 

    tol = 1e-12;
    if (abs(a+1/2.)<tol) & (abs(b+1/2.)<tol) :
        return cheb1.glquad(N,shift,scale)
    else :
        [a_s,b_s] = recurrence(N,a,b,shift,scale)
        return opoly1.opoly_glq(a_s,b_s,N,r0=r0)
