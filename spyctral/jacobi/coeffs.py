#! /usr/bin/env python
# 
# Functions for evaluating coefficients regarding Jacobi polynomials

__all__ = ['recurrence_range',]

def recurrence_range(N,alpha=-1/2.,beta=-1/2.):
# Returns the first N Jacobi recurrence coefficients

    return recurrence(range(N),alpha=alpha,beta=beta)

def recurrence(ns,alpha=-1/2.,beta=-1/2.):
# Returns recurrence coefficients for certain values of n
    from numpy import array, zeros, ones, any
    from scipy.special import gamma

    if type(ns) != list:
        ns = [ns]
    ns = array(ns)
    alpha = float(alpha)
    beta = float(beta)
    a_s = (beta**2-alpha**2)*ones(ns.shape)
    b_s = zeros(ns.shape)

    flags0 = ns==0
    if any(flags0):
        a_s[flags0] = (beta-alpha)/(alpha+beta+2)
        b_s[flags0] = 2**(alpha+beta+1)*gamma(alpha+1)*\
                     gamma(beta+1)/gamma(alpha+beta+2)
    flags1 = ns==1
    if any(flags1):
        a_s[flags1] /= (2+alpha+beta)*(2+alpha+beta+2)
        b_s[flags1] = 4*(alpha+1)*(beta+1)/((2+alpha+beta)**2*(3+alpha+beta))

    flags = ~(flags0 | flags1)
    n = ns[flags]
    if any(flags):
        temp = 2*n+alpha+beta
        num = 4*n*(n+alpha)*(n+beta)*(n+alpha+beta)
        den = (temp-1)*(temp**2)*(temp+1)
        b_s[flags] = num/den
        a_s[flags] /= temp*(temp+2)

    return [a_s,b_s]

def zetan(n,alpha=-1/2.,beta=-1/2.,normalization='normal'):
# Computes the derivative coefficient eta for the normalized polynomials:
# d/dr P_n^(a,b) = zeta*P_{n-1}^(a+1,b+1)
# zeta depends on the normalization
    from numpy import sqrt

    if normalization == 'normal':
        return sqrt(n*(n+alpha+beta+1))

# Compute the coefficients expanding (1-r**2) to the next lower Jacobi class
# (1-r**2)*P_n^(a,b) = e_0*P_n^(a-1,b-1) + e_1*P_{n+1}^(a-1,b-1) + 
#                      e_2*P_{n+2}^(a-1,b-1)
def epsilonn(n,alpha=1/2.,beta=1/2.):

    from numpy import arrray, zeros, sqrt
    n = array(n)
    N = n.size
    n = n.reshape(N)

    a = alpha
    b = beta

    epsn = zeros([N,3])
    epsn[:,0] = sqrt(4*(n+a)*(n+b)*(n+a+b-1)*(n+a+b)/ \
                        ((2*n+a+b-1)*(2*n+a+b)**2*(2*n+a+b+1)))
    epsn[:,1] = 2*(alpha-beta)*sqrt((n+1)*(n+a+b))/ \
                ((2*n+a+b)*(2*n+a+b+2))
    epsn[:,2] = -sqrt(4*(n+1)*(n+2)*(n+a+1)*(n+b+1)/ \
                           ((2*n+a+b+1)*(2*n+a+b+2)**2*(2*n+a+b+3)))
    
    return epsn.squeeze()

# Coefficients for expanding (a,b) polynomial to (a+1,b+1) polynomial
# P_n^(a,b) = h_2*P_n^(a+1,b+1) + h_1*P_{n-1}^(a+1,b+1) + h_0*P_{n-2}^(a+1,b+1)
def etan(n,alpha=-1/2.,beta=-1/2.):

    from numpy import array, zeros, sqrt
    n = array(n)
    N = n.size
    n = n.reshape(N)

    a = alpha
    b = beta

    etas = zeros([N,3])

    num = 4*(n+a+1)*(n+b+1)*(n+a+b+1)*(n+a+b+2)
    temp = (2*n+a+b)
    den = (temp+1)*((temp+2)**2)*(temp+3)
    etas[:,2] = sqrt(num/den)
    etas[:,1] = 2*(a-b)*sqrt(n*(n+a+b+1))/(temp*(temp+2))
    
    num = 4*n*(n-1)*(n+a)*(n+b)
    den = (temp-1)*(temp**2)*(temp+1)
    etas[:,0] = -sqrt(num/den)

    return etas.squeeze()

# Coefficients for promoting polynomial (a,b) to (a+1,b) or (a,b+1)
# P_n^(a,b) = -delta^(a,b)_0*P_{n-1}^(a+1,b) + delta^(a,b)_1*P_n^(a+1,b)
# P_n^(a,b) = delta^(b,a)_0*P_{n-1}^(a,b+1) + delta^(b,a)_1*P_n^(a,b+1)
def deltan(n,alpha=-1/2.,beta=-1/2.):
    from numpy import array, zeros, sqrt

    n = array(n)
    N = n.size
    n = n.reshape(N)

    a = alpha
    b = beta

    deltas = zeros([N,2])

    deltas[:,0] = 2*n*(n+b)/((2*n+a+b)*(2*n+a+b+1))
    deltas[:,1] = 2*(n+a+1)*(n+a+b+1)/((2*n+a+b+1)*(2*n+a+b+2))
    deltas = sqrt(deltas)

    return deltas.squeeze()

# Coefficients for demoting polynomial (a,b) to (a-1,b) or (a,b-1)
# (1-r)*P_n^(a,b) = gamma^(a,b)_0*P_{n}^(a-1,b) - gamma^(a,b)_1*P_{n+1}^(a-1,b)
# (1+r)*P_n^(a,b) = gamma^(b,a)_0*P_{n}^(a,b-1) + gamma^(b,a)_1*P_{n+1}^(a,b-1)
def gamman(n,alpha=1/2.,beta=1/2.):
    from numpy import array, zeros, sqrt

    n = array(n)
    N = n.size
    n = n.reshape(N)

    a = alpha
    b = beta

    gammas = zeros([N,2])

    temp = 2*n+a+b
    gammas[:,0] = 2*(n+a)*(n+a+b)/(temp*(temp+1))
    gammas[:,1] = 2*(n+1)*(n+b+1)/((temp+1)*(temp+2))
    gammas = sqrt(gammas)

    return gammas.squeeze()
