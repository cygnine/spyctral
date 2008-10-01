#! /usr/bin/env python
# 
# Functions for evaluation and quadrature of Jacobi polynomials

__all__ = ['recurrence','jpoly','jpolyn','gq']

import numpy as _np
import scipy.special as _nf

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

from opoly1 import eval_opoly, eval_opolyn, opoly_gq

def jpoly(x,n,alpha=-1/2.,beta=-1/2.,d=0) :
    N = _np.max(n);
    [a,b] = recurrence(N+1,alpha,beta)
    return eval_opoly(x,n,a,b,d)

def jpolyn(x,n,alpha=-1/2.,beta=-1/2.,d=0) :
    N = _np.max(n);
    [a,b] = recurrence(N+2,alpha,beta)
    return eval_opolyn(x,n,a,b,d)

# Returns the N-point Jacobi-Gauss(a,b) quadrature rule over the interval
# (-scale,scale)+shift
def gq(N,a=-1/2.,b=-1/2.,shift=0,scale=1) : 
    [a_s,b_s] = recurrence(N,a,b,shift,scale)
    return opoly_gq(a_s,b_s,N)
