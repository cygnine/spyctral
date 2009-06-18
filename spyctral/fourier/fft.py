# Module to implement built-in numpy fft for computing fourier modes
# 
# 20090302 -- acn

__all__ = []

# Uses numpy fft to obtain modes. Only works on 1D arrays
def fft(fx,g=0,d=0,scale=1):
    from numpy import sqrt, roll, arange, exp, sign
    from numpy.fft import fft as ft
    from scipy import pi
    from quad import N_to_ks

    from connection import int_connection

    N = fx.size
    modes = ft(fx)*sqrt(2*pi)/N

    ks = N_to_ks(N)

    modes = roll(modes,N/2)

    flags = ks!=0
    flags2 = ((ks%2)==1)
    modes[flags] *= exp(-1j*ks[flags]*pi/N)
    modes[flags2] *= -1

    return int_connection(modes,0,0,g,d)

# Uses numpy ifft to obtain nodes. Only works on 1D arrays
# Just reverses what was done in fft above
def ifft(f,g=0.,d=0.,scale=1.):
    from numpy import sqrt, roll, arange, exp, sign
    from numpy.fft import ifft as ift
    from scipy import pi
    from quad import N_to_ks
    from connection import int_connection_backward

    N = f.size
    ks = N_to_ks(N)
    fx = int_connection_backward(f,0,0,g,d)

    flags = ks!=0
    flags2 = ((ks%2)==1)

    fx[flags2] *= -1
    fx[flags] /= exp(-1j*ks[flags]*pi/N)

    fx = roll(fx,-(N-1)/2)

    return N/sqrt(2*pi)*ift(fx)

# Combines overhead calculations needed to perform fft
def fft_overhead(N,g=0.,d=0.,scale=1.):
    from numpy import sqrt, arange, exp, sign,ones,array
    from scipy import pi
    from quad import N_to_ks

    from connection import int_connection_overhead

    ks = N_to_ks(N)

    flags = ks!=0
    flags2 = ((ks%2)==1)

    factors = exp(-1j*ks[flags]*pi/N)
    factors2 = -1

    all_factors = ones(N,dtype='complex128')
    all_factors[flags] *= factors
    all_factors[flags2] *= factors2
    all_factors *= sqrt(2*pi)/N

    matrices = int_connection_overhead(N,0,0,g,d)

    return [all_factors,matrices]

# Does the direct FFT using the overhead calculations from fft_overhead
def fft_online(fx,overhead):
    from numpy.fft import fft as ft
    from numpy import roll
    from connection import int_connection_online as int_connection

    modes = ft(fx)
    N = fx.size

    modes = roll(modes,N/2)

    modes *= overhead[0]

    return int_connection(modes,overhead[1])

# Defines ifft_online: the overhead necessary is computed by
# fft_overhead: instead of multiplying it, we'll just divide by it.
def ifft_online(f,overhead):

    from numpy.fft import ifft as ift
    from numpy import roll

    from connection import int_connection_backward_online

    fx = int_connection_backward_online(f,overhead[1])
    fx /= overhead[0]
    N = fx.size

    fx = roll(fx,-(N/2))

    return ift(fx)
