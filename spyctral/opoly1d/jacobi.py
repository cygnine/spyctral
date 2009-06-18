#! /usr/bin/env python
# 
# Functions for evaluation and quadrature of Jacobi polynomials

__all__ = ['recurrence','jpoly','jpolyn','gquad','glquad', 'grquad',
        'gamma']

import numpy as _np
import scipy.special as _nf
#from opoly1 import eval_opoly, eval_opolyn, opoly_gq, opoly_grq, opoly_glq
import opoly1
import cheb1
import spectral_common

Decimate = spectral_common.DecimateN
LowPass = spectral_common.LowPassN

# Returns the first N Jacobi recurrence coefficients
def recurrence(N,alpha=-1/2.,beta=-1/2.,shift=0,scale=1) :

    from opoly1.eval import recurrence_scaleshift

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

    # Deprecate recurrence_scaleshift: easier to just do affine
    # transformations
    #return recurrence_scaleshift([a_s,b_s],scale=scale,shift=shift)
    return [a_s,b_s]
        
    # Still have recurrence_scaleshift to deal with

# Returns recurrence coefficients for certain values of n
def recurrence_ns(ns,alpha=-1/2.,beta=-1/2.,shift=0,scale=1) :

    from opoly1.eval import recurrence_scaleshift

    ns = _np.array(ns)
    N = ns.size
    ns = ns.reshape([N])
    alpha = float(alpha)
    beta = float(beta)
    a_s = (beta**2-alpha**2)*_np.ones([N])
    b_s = _np.zeros([N])

    count = 0
    for n in ns:
        if n==0:
            a_s[count] = (beta-alpha)/(alpha+beta+2);
            b_s[count] = 2**(alpha+beta+1)*_nf.gamma(alpha+1)*_nf.gamma(beta+1)/_nf.gamma(alpha+beta+2)
        elif n==1:
            a_s[count] = a_s[count]/((2*n+alpha+beta)*(2*n+alpha+beta+2))
            b_s[count] = 4*n*(n+alpha)*(n+beta)/((2*n+alpha+beta)**2*(2*n+alpha+beta+1))
        else: 
            a_s[count] = a_s[count]/((2*n+alpha+beta)*(2*n+alpha+beta+2))
            num = 4*n*(n+alpha)*(n+beta)*(n+alpha+beta)
            den =  (2*n+alpha+beta)**2*(2*n+alpha+beta+1)*(2*n+alpha+beta-1)
            b_s[count] = num/den
        count += 1

    # Deprecate recurrence_scaleshift: easier to just do affine
    # transformations
    #return recurrence_scaleshift([a_s,b_s],scale=scale,shift=shift)
    return [a_s,b_s]


# Evaluates the monic Jacobi polynomials of class (alpha,beta), order n (list)
# at the points x (list)
def jpoly(x,n,alpha=-1/2.,beta=-1/2.,d=0, scale=1., shift=0.) :
    from numpy import arange
    from spectral_common import forward_scaleshift as fss
    from spectral_common import backward_scaleshift as bss
    N = _np.max(n);
    [a,b] = recurrence(N+1,alpha,beta)

    # Shift to standard interval and use opoly 3-term recurrence
    fss(x,scale=scale,shift=shift)
    temp = opoly1.eval_opoly(x,n,a,b,d)
    bss(x,scale=scale,shift=shift)

    # Scale appropriately (monic is baaaad if scale is large)
    temp = temp*(scale**arange(N))
    
    return temp

# Evaluates the L^2-normalized Jacobi polynomials of class (alpha,beta), order n (list)
# at the points x (list)
def jpolyn(x,n,alpha=-1/2.,beta=-1/2.,d=0,scale=1.,shift=0.) :

    from numpy import arange,sqrt
    from spectral_common import forward_scaleshift as fss
    from spectral_common import backward_scaleshift as bss

    n = _np.array(n)
    N = _np.max(n);
    [a,b] = recurrence(N+2,alpha,beta)

    # Shift to standard interval and use opoly 3-term recurrence
    fss(x,scale=scale,shift=shift)
    temp = opoly1.eval_opolyn(x,n,a,b,d)
    bss(x,scale=scale,shift=shift)

    # Scale appropriately
    temp /= sqrt(scale)

    return temp

# Temporary function to evaluate Jacobi derivatives
def djpolyn(x,n,alpha=-1/2.,beta=-1/2.,scale=1.,shift=0.):
    N = _np.max(n)
    n = _np.array(n)
    zetas = zetan(n,alpha=alpha,beta=beta)/scale
    #temp = _np.diag(_np.sqrt(n*(n+alpha+beta+1)))
    return jpolyn(x,n-1,alpha+1.,beta+1.,shift=shift,scale=scale)*zetas
    #return _np.dot(jpolyn(x,n-1,alpha+1.,beta+1.),temp)

# Returns the N-point Jacobi-Gauss(a,b) quadrature rule over the interval
# (-scale,scale)+shift
# The quadrature rule is normalized to reflect the real Jacobian
def gquad(N,a=-1/2.,b=-1/2.,shift=0.,scale=1.) : 
    from spectral_common import backward_scaleshift as bss

    tol = 1e-12;
    if (abs(a+1/2.)<tol) & (abs(b+1/2.)<tol) :
        return cheb1.gquad(N,shift=shift,scale=scale)
    else :
        [a_s,b_s] = recurrence(N,a,b)
        temp = opoly1.opoly_gq(a_s,b_s,N)
        temp[1] *= scale
        bss(temp[0],scale=scale,shift=shift)
        return temp

# Returns the N-point Jacobi-Gauss-Radau(a,b) quadrature rule over the interval
# (-scale,scale)+shift
def grquad(N,a=-1/2.,b=-1/2.,r0=-1.,shift=0,scale=1) : 
    from spectral_common import forward_scaleshift as fss
    from spectral_common import backward_scaleshift as bss

    tol = 1e-12;
    if (abs(a+1/2.)<tol) & (abs(b+1/2.)<tol) & (abs(abs(r0)-1.)<tol) :
        return cheb1.grquad(N,r0=r0,shift=shift,scale=scale)
    else :
        [a_s,b_s] = recurrence(N,a,b,shift,scale)
        fss(r0,scale=scale,shift=shift)
        temp = opoly1.opoly_gq(a_s,b_s,N,r0=r0)
        bss(r0,scale=scale,shift=shift)
        bss(temp[0],scale=scale,shift=shift)
        temp[1] *= scale
        return temp

# Returns the N-point Jacobi-Gauss-Lobatto(a,b) quadrature rule over the interval
# (-scale,scale)+shift
def glquad(N,a=-1/2.,b=-1/2.,r0=False,shift=0.,scale=1.) : 
    from spectral_common import forward_scaleshift as fss
    from spectral_common import backward_scaleshift as bss

    from numpy import array

    if type(r0)==bool:
        r0 = [-scale,scale]

    r0 = array(r0)
    tol = 1e-12;
    if (abs(a+1/2.)<tol) & (abs(b+1/2.)<tol) :
        return cheb1.glquad(N,shift,scale)
    else :
        [a_s,b_s] = recurrence(N,a,b)
        fss(r0,scale=scale,shift=shift)
        temp = opoly1.opoly_glq(a_s,b_s,N,r0=r0)
        bss(r0,scale=scale,shift=shift)
        bss(temp[0],scale=scale,shift=shift)
        temp[1] *= scale
        return temp

########################################################
#                 MATRIX FUNCTIONS                     #
########################################################

# Applies the modal stiffness matrix to the coefficients of the L2
# normalized polynomials. Makes use of the recurrence constant zetan in
# addition to the sparse representation of the connection coefficients
# Is an O(N) operation
def stiff_apply(F,alpha=-1/2.,beta=-1/2.,scale=1.):
    from jfft import rmatrix_invert
    from numpy import arange,hstack, array

    N = F.size
    # Input F is of class (alpha,beta). Take the derivative by promoting
    # basis functions to (alpha+1,beta+1) using zetan:
    zetas = zetan(arange(N),alpha=alpha,beta=beta)/scale
    # Now demote the (alpha+1,beta+1) coefficients back down to
    # (alpha,beta)
    filler = array([0.])
    return hstack((rmatrix_invert(zetas[1:]*F[1:],alpha,beta,1,1),filler))

# Calculates overhead required for applying the modal stiffness matrix
def stiff_overhead(N,alpha=-1/2.,beta=-1/2.,scale=1.):
    from jfft import rmatrix_entries
    from numpy import arange

    zetas = zetan(arange(N),alpha=alpha,beta=beta)/scale
    Rs = rmatrix_entries(N-1,alpha,beta,1,1)
    return [zetas[1:],Rs]

# Performs the online application of the stiffness matrix given the
# overhead from stiff_overhead as input
def stiff_online(F,overhead):
    from jfft import rmatrix_entries_invert
    from numpy import array,hstack

    filler = array([0.])
    return hstack((rmatrix_entries_invert(overhead[0]*F[1:],overhead[1]),filler))


########################################################
#                 HELPER FUNCTIONS                     #
########################################################

# Given a function, computes modes of the function in a certain class expansion
# using the Gauss quadrature native to that expansion. Computes the first N
# modes of f in the (a,b) expansion using (a,b) Gauss quadrature
def expand_gq(f,N,a,b):

    [r,w] = gquad(N,a,b)
    r = r.squeeze()
    w = w.squeeze()
    ps = jpolyn(r,range(N),a,b)
    return _np.dot(ps.T,w*f(r))


########################################################
#                 SECONDARY COEFFICIENTS               #
########################################################

# Computes the connection coefficients transferring (a,b) Jacobi polynomials
# into (c,d) Jacobi polynomials. This is done numerically via quadrature, which
# is analytically exact.
def conncoeff_num(N,a,b,c,d):

    # Get quadrature FINISH
    return False

# Computes the derivative coefficient eta for the normalized polynomials:
# d/dr P_n^(a,b) = zeta*P_{n-1}^(a+1,b+1)
def zetan(n,alpha=-1/2.,beta=-1/2.):
    return _np.sqrt(n*(n+alpha+beta+1))

# Compute the coefficients expanding (1-r**2) to the next lower Jacobi class
# (1-r**2)*P_n^(a,b) = e_0*P_n^(a-1,b-1) + e_1*P_{n+1}^(a-1,b-1) + 
#                      e_2*P_{n+2}^(a-1,b-1)
def epsilonn(n,alpha=1/2.,beta=1/2.):

    n = _np.array(n)
    N = n.size
    n = n.reshape(N)

    a = alpha
    b = beta

    epsn = _np.zeros([N,3])
    epsn[:,0] = _np.sqrt(4*(n+a)*(n+b)*(n+a+b-1)*(n+a+b)/ \
                        ((2*n+a+b-1)*(2*n+a+b)**2*(2*n+a+b+1)))
    epsn[:,1] = 2*(alpha-beta)*_np.sqrt((n+1)*(n+a+b))/ \
                ((2*n+a+b)*(2*n+a+b+2))
    epsn[:,2] = -_np.sqrt(4*(n+1)*(n+2)*(n+a+1)*(n+b+1)/ \
                           ((2*n+a+b+1)*(2*n+a+b+2)**2*(2*n+a+b+3)))
    
    return epsn.squeeze()

# Coefficients for expanding (a,b) polynomial to (a+1,b+1) polynomial
# P_n^(a,b) = h_2*P_n^(a+1,b+1) + h_1*P_{n-1}^(a+1,b+1) + h_0*P_{n-2}^(a+1,b+1)
def etan(n,alpha=-1/2.,beta=-1/2.):

    n = _np.array(n)
    N = n.size
    n = n.reshape(N)

    a = alpha
    b = beta

    etas = _np.zeros([N,3])

    num = 4*(n+a+1)*(n+b+1)*(n+a+b+1)*(n+a+b+2)
    temp = (2*n+a+b)
    den = (temp+1)*((temp+2)**2)*(temp+3)
    etas[:,2] = _np.sqrt(num/den)
    etas[:,1] = 2*(a-b)*_np.sqrt(n*(n+a+b+1))/(temp*(temp+2))
    
    num = 4*n*(n-1)*(n+a)*(n+b)
    den = (temp-1)*(temp**2)*(temp+1)
    etas[:,0] = -_np.sqrt(num/den)

    return etas.squeeze()

# Coefficients for promoting polynomial (a,b) to (a+1,b) or (a,b+1)
# P_n^(a,b) = -delta^(a,b)_0*P_{n-1}^(a+1,b) + delta^(a,b)_1*P_n^(a+1,b)
# P_n^(a,b) = delta^(b,a)_0*P_{n-1}^(a,b+1) + delta^(b,a)_1*P_n^(a,b+1)
def deltan(n,alpha=-1/2.,beta=-1/2.):

    n = _np.array(n)
    N = n.size
    n = n.reshape(N)

    a = alpha
    b = beta

    deltas = _np.zeros([N,2])

    deltas[:,0] = 2*n*(n+b)/((2*n+a+b)*(2*n+a+b+1))
    deltas[:,1] = 2*(n+a+1)*(n+a+b+1)/((2*n+a+b+1)*(2*n+a+b+2))
    deltas = _np.sqrt(deltas)

    return deltas.squeeze()

# Coefficients for demoting polynomial (a,b) to (a-1,b) or (a,b-1)
# (1-r)*P_n^(a,b) = gamma^(a,b)_0*P_{n}^(a-1,b) - gamma^(a,b)_1*P_{n+1}^(a-1,b)
# (1+r)*P_n^(a,b) = gamma^(b,a)_0*P_{n}^(a,b-1) + gamma^(b,a)_1*P_{n+1}^(a,b-1)
def gamman(n,alpha=1/2.,beta=1/2.):

    n = _np.array(n)
    N = n.size
    n = n.reshape(N)

    a = alpha
    b = beta

    gammas = _np.zeros([N,2])

    temp = 2*n+a+b
    gammas[:,0] = 2*(n+a)*(n+a+b)/(temp*(temp+1))
    gammas[:,1] = 2*(n+1)*(n+b+1)/((temp+1)*(temp+2))
    gammas = _np.sqrt(gammas)

    return gammas.squeeze()


########################################################
#                 NODAL FUNCTIONS                      #
########################################################
def scale_nodes(L,N,delta=0.5,alpha=-0.5,beta=-0.5):

    from spectral_common import scale_factor

    x = gquad(N,a=alpha,b=beta)[0]

    return scale_factor(L,x,scale=delta)
