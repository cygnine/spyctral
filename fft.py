# Module to implement built-in numpy fft for computing fourier modes
# 
# 20090302 -- acn

__all__ = []

# Uses numpy fft to obtain modes. Only works on 1D arrays
# DOES NOT DO G,D != 0 YET
def fft(fx,g=0,d=0):
    from numpy import sqrt, hstack, arange, exp, sign
    from numpy.fft import fft as ft
    from scipy import pi

    N = fx.size
    modes = ft(fx)*sqrt(2*pi)/N

    if bool(N%2):
        n = (N-1)/2
        modes = hstack((modes[-n:],modes[:(n+1)]))
        ks = arange(-n,n+1)
    else:
        n = N/2
        modes = hstack((modes[-n:],modes[:n]))
        ks = arange(-n,n)

    flags = ks!=0
    flags2 = ((ks%2)==1)
    modes[flags] *= exp(-1j*ks[flags]*pi/N)
    modes[flags2] *= -1

    return modes

# Combines overhead calculations needed to perform fft
def fft_overhead(N,g=0.,d=0.):
    from numpy import sqrt, arange, exp, sign,ones,array
    from scipy import pi
    from quad import N_to_ks

    ks = N_to_ks(N)

    flags = ks!=0
    flags2 = ((ks%2)==1)

    factors = exp(-1j*ks[flags]*pi/N)
    factors2 = -1

    all_factors = ones(N,dtype='complex128')
    all_factors[flags] *= factors
    all_factors[flags2] *= factors2
    all_factors *= sqrt(2*pi)/N

    return all_factors

# Does the direct FFT using the overhead calculations from fft_overhead
def fft_online(fx,overhead,g=0.,d=0.):
    from numpy.fft import fft as ft
    from numpy import hstack

    modes = ft(fx)
    N = fx.size
    if bool(N%2):
        n = (N-1)/2
        modes = hstack((modes[-n:],modes[:(n+1)]))
    else:
        n = N/2
        modes = hstack((modes[-n:],modes[:n]))

    modes *= overhead

    return modes

# Uses numpy ifft to obtain nodes. Only works on 1D arrays
# Just reverses what was done in fft above
# DOES NOT DO G,D != 0 YET
def ifft(f,g=0.,d=0.):
    from numpy import sqrt, hstack, arange, exp, sign
    from numpy.fft import ifft as ift
    from scipy import pi
    from quad import N_to_ks

    N = f.size
    ks = N_to_ks(N)
    fx = f.copy()

    flags = ks!=0
    flags2 = ((ks%2)==1)

    fx[flags2] *= -1
    fx[flags] /= exp(-1j*ks[flags]*pi/N)

    if bool(N%2):
        n = (N-1)/2
        fx = hstack((fx[-(n+1):],fx[:n]))
    else:
        n = N/2
        fx = hstack((fx[-n:],fx[:n]))

    return N/sqrt(2*pi)*ift(fx)

# Defines ifft_online: the overhead necessary is computed by
# fft_overhead: instead of multiplying it, we'll just divide by it.
def ifft_online(f,overhead,g=0.,d=0.):

    from numpy.fft import ifft as ift
    from numpy import hstack

    fx = f/overhead
    N = fx.size
    if bool(N%2):
        n = (N-1)/2
        fx = hstack((fx[-(n+1):],fx[:n]))
    else:
        n = N/2
        fx = hstack((fx[-n:],fx[:n]))

    return ift(fx)
