#!/usr/bin/env python

# FFT module for mapped Jacobi polynomials
# NOTE: the tranformation from x \in \mathbb{R} to r \in [-1,1] does not
# preserve affine scalings. I.e., the scaling parameter on the infinite
# interval is not the same as the scaling parameter on the finite
# interval. Because of this, the scalings on the infinite interval are
# just applied after all the finite interal stuff (and not built-in to
# the finite-interval stuff).

__all__ = []

# Performs the FFT for the *weighted* mapped Jacobi polynomials when s,t
# are integers.
def mjwfft(u,s=1.,t=1.,shift=0.,scale=1.):
    from jfft import jacfft
    from maps import sqrt_wjacobiw
    from quad import gq

    A = s-1
    B = t-1

    A = int(A)
    B = int(B)

    N = u.size

    # Use canonical quadrature points
    [x,w] = gq(N,s=1.,t=1.,scale=scale,shift=shift)

    # Must multiply by the sqrt of the Jacobian and then do FFT
    factors = sqrt_wjacobiw(x,s=s,t=t,scale=scale,shift=shift)
    return jacfft(u/factors,A,B)

# Offline precomputations for the FFT
def mjwfft_overhead(N,s=1.,t=1.,shift=0.,scale=1.):
    from jfft import jacfft_overhead
    from maps import sqrt_wjacobiw
    from quad import gq
    
    A = s-1
    B = t-1

    A = int(A)
    B = int(B)

    # Use canonical quadrature points
    [x,w] = gq(N,s=1.,t=1.,scale=scale,shift=shift)

    # Must multiply by the sqrt of the Jacobian and then do FFT
    factors = sqrt_wjacobiw(x,s=s,t=t,scale=scale,shift=shift)

    jo = jacfft_overhead(N,A,B)
    return [factors, jo]

# Online (fast) portion of fft
# overhead[0]: multiplicative factor of sqrt_wjacobiw
# overhead[1]: overhead for jfft.jacfft
def mjwfft_online(u,overhead):
    from jfft import jacfft_online as fft

    return fft(u/overhead[0],overhead[1])

# Performs the IFFT for the *weighted* mapped Jacobi polynomials when s,t
# are integers.
def mjwifft(U,s=1.,t=1.,shift=0.,scale=1.):
    from jfft import jacifft
    from maps import sqrt_wjacobiw
    from quad import gq

    A = s-1
    B = t-1

    A = int(A)
    B = int(B)

    N = U.size

    # Use canonical quadrature points
    [x,w] = gq(N,s=1.,t=1.,scale=scale,shift=shift)

    # Inverse transform
    u = jacifft(U,A,B)

    # Multiply by factors
    factors = sqrt_wjacobiw(x,s=s,t=t,scale=scale,shift=shift)
    return u*factors

# Online (fast) portion of ifft
# overhead[0]: multiplicative factor of sqrt_wjacobiw
# overhead[1]: overhead for jfft.jacifft
def mjwifft_online(u,overhead):
    from jfft import jacifft_online as ifft

    return overhead[0]*ifft(u,overhead[1])
