# Module for the evaluation of the rational Wiener functions

#import maps

#__all__ = ['genwiener',
#           'genwienerw']

# Evaluates the orthonormalized unweighted Wiener rational functions
def wiener(x,k,s=1.,t=0.,shift=0.,scale=1.):

    from numpy import sqrt, array
    from spyctral.fourier.eval import fseries as genfourier
    from spyctral.wiener.maps import x_to_theta as x2theta

    # Preprocessing and setup
    x = array(x)
    x = x.ravel()
    k = array(k,dtype=int)
    k = k.ravel()

    theta = x2theta(x,shift=shift,scale=scale)

    return genfourier(theta,k,s-1.,t)/sqrt(scale)

# Evaluates the derivative of the orthonormalized unweighted Wiener rational
# functions
def dwiener(x,k,s=1.,t=0.):

    from spyctral.fourier.eval import dfseries as dgenfourier
    from spyctral.wiener.maps import x_to_theta as x2theta
    from spyctral.wiener.maps import x_to_r as x2r
    from numpy import array

    # Preprocessing and setup
    x = array(x)
    x = x.ravel()
    k = array(k,dtype=int)
    k = k.ravel()   

    theta = x2theta(x,shift=shift,scale=scale)
    r = x2r(x,shift=shift,scale=scale)

    return ((1+r)*(dgenfourier(theta,k,s-1.,t).T)).T/scale

# Evaluates the orthonormalized weighted Wiener rational functions
def weighted_wiener(x,k,s=1.,t=0.,shift=0.,scale=1.):

    from numpy import sqrt, array
    from spyctral.fourier.eval import fseries as genfourier
    from spyctral.wiener.maps import x_to_theta as x2theta
    from spyctral.wiener.weights import sqrt_weight_bias as wx_sqrt

    # Preprocessing and setup
    x = array(x)
    x = x.ravel()
    k = array(k,dtype=int)
    k = k.ravel()

    theta = x2theta(x,shift=shift,scale=scale)

    psi = genfourier(theta,k,s-1.,t)

    psi = (wx_sqrt((x-shift)/scale,s,t)*(psi.T)).T

    return psi/sqrt(scale)

# Evaluates the derivative of the orthonormalized weighted Wiener rational functions
def dweighted_wiener(x,k,s=1.,t=0.,shift=0.,scale=1.):

    from numpy import sqrt, array
    from spyctral.fourier.eval import fseries as genfourier
    from spyctral.fourier.eval import dfseries as dgenfourier
    from spyctral.wiener.maps import dtheta_dx
    from spyctral.wiener.maps import x_to_theta as x2theta
    from spyctral.wiener.weights import sqrt_weight_bias as wx_sqrt
    from spyctral.wiener.weights import dsqrt_weight_bias as dwx_sqrt

    # Preprocessing and setup
    x = array(x)
    x = x.ravel()
    k = array(k,dtype=int)
    k = k.ravel()

    theta = x2theta(x,shift=shift,scale=scale)

    # First term: wx_sqrt * d/dx Phi
    psi = (dtheta_dx(x,shift=shift,scale=scale)*dgenfourier(theta,k,s-1.,t).T).T
    psi = (wx_sqrt(x,s,t,shift=shift,scale=scale)*(psi.T)).T
    
    # Second term: d/dx wx_sqrt * Phi
    psi += (dwx_sqrt(x,s,t,shift=shift,scale=scale)*genfourier(theta,k,s-1.,t).T).T

    return psi/sqrt(scale)


# Evaluates the orthonormalized weighted Wiener rational functions
def xiw(x,n,s=1.,t=0.,shift=0.,scale=1.):

    from numpy import sqrt, array
    from spyctral.fourier.eval import fseries as genfourier
    from spyctral.wiener.weights import sqrt_weight_bias as wx_sqrt
    from spyctral.wiener.maps import x_to_theta as x2theta

    # Preprocessing and setup
    x = array(x)
    x = x.ravel()
    n = array(n,dtype=int)
    n = n.ravel()

    theta = x2theta(x,shift=shift,scale=scale)

    psi = genfourier(theta,n,s-1.,t).real

    psi = (wx_sqrt((x-shift)/scale,s,t)*(psi.T)).T

    psi[:,n==0] *= sqrt(2)
    psi[:,n!=0] *= 2

    return psi/sqrt(scale)


# Evaluates the derivative of the orthonormalized weighted Wiener rational functions
def dxiw(x,n,s=1.,t=0.,shift=0.,scale=1.):

    from numpy import sqrt, array
    from spyctral.fourier.eval import fseries as genfourier
    from spyctral.fourier.eval import dfseries as dgenfourier
    from spyctral.wiener.maps import dtheta_dx
    from spyctral.wiener.maps import x_to_theta as x2theta
    from spyctral.wiener.weights import sqrt_weight_bias as wx_sqrt
    from spyctral.wiener.weights import dsqrt_weight_bias as dwx_sqrt

    # Preprocessing and setup
    x = array(x)
    x = x.ravel()
    n = array(n,dtype=int)
    n = n.ravel()

    theta = x2theta(x,shift=shift,scale=scale)

    # First term: wx_sqrt * d/dx Phi
    psi = (dthetadx(x,shift=shift,scale=scale)*dgenfourier(theta,n,s-1.,t).real.T).T
    psi = (wx_sqrt(x,s,t,shift=shift,scale=scale)*(psi.T)).T
    
    # Second term: d/dx wx_sqrt * Phi
    psi += (dwx_sqrt(x,s,t,shift=shift,scale=scale)*genfourier(theta,n,s-1.,t).real.T).T

    psi[:,n==0] *= sqrt(2)
    psi[:,n!=0] *= 2

    return psi/sqrt(scale)
