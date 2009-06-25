# Module to implement built-in numpy fft for computing fourier modes
# 
# 20090302 -- acn

__all__ = []

# Uses numpy fft to obtain modes. Only works on 1D arrays. gamma and delta must
# be integers
def fft(fx,gamma=0,delta=0,scale=1):
    from numpy import sqrt, roll, arange, exp, sign
    from numpy.fft import fft as ft
    import spyctral.common.fft as pyfft
    from scipy import pi
    from spyctral.common.indexing import integer_range

    from connection import int_connection

    N = fx.size
    if fx.dtype is object:
        modes = pyfft.fft(fx)*sqrt(2*pi)/N
    else:
        modes = ft(fx)*sqrt(2*pi)/N

    ks = integer_range(N)

    modes = roll(modes,N/2)

    flags = ks!=0
    flags2 = ((ks%2)==1)
    modes[flags] *= exp(-1j*ks[flags]*pi/N)
    modes[flags2] *= -1

    return int_connection(modes,0,0,gamma,delta)

# Uses numpy ifft to obtain nodes. Only works on 1D arrays
# Just reverses what was done in fft above
def ifft(f,gamma=0.,delta=0.,scale=1.):
    from numpy import sqrt, roll, arange, exp, sign
    from numpy.fft import ifft as ift
    import spyctral.common.fft as pyfft
    from scipy import pi
    from spyctral.common.indexing import integer_range
    from connection import int_connection_backward

    N = f.size
    ks = integer_range(N)
    fx = int_connection_backward(f,0,0,gamma,delta)

    flags = ks!=0
    flags2 = ((ks%2)==1)

    fx[flags2] *= -1
    fx[flags] /= exp(-1j*ks[flags]*pi/N)

    fx = roll(fx,-(N-1)/2)

    if fx.dtype is object:
        return N/sqrt(2*pi)*ift(fx)
    else:
        return 1/sqrt(2*pi)*pyfft.fft(fx,sign=-1)


# Combines overhead calculations needed to perform fft
def fft_overhead(N,gamma=0.,delta=0.,scale=1.):
    from numpy import sqrt, arange, exp, sign,ones,array
    from scipy import pi
    from spyctral.common.indexing import integer_range 

    from connection import int_connection_overhead

    ks = integer_range(N)

    flags = ks!=0
    flags2 = ((ks%2)==1)

    factors = exp(-1j*ks[flags]*pi/N)
    factors2 = -1

    all_factors = ones(N,dtype=complex)
    all_factors[flags] *= factors
    all_factors[flags2] *= factors2
    all_factors *= sqrt(2*pi)/N

    matrices = int_connection_overhead(N,0,0,gamma,delta)

    return [all_factors,matrices]

# Does the direct FFT using the overhead calculations from fft_overhead
def fft_online(fx,overhead):
    from numpy.fft import fft as ft
    from numpy import roll
    import spyctral.common.fft as pyfft
    from connection import int_connection_online as int_connection

    if fx.dtype is object:
        modes = pyfft.fft(fx)
    else:
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
    import spyctral.common.fft as pyfft

    from connection import int_connection_backward_online

    fx = int_connection_backward_online(f,overhead[1])
    fx /= overhead[0]
    N = fx.size

    fx = roll(fx,-(N/2))

    if fx.dtype is object:
        return pyfft.fft(fx,sign=-1)/N
    else:
        return ift(fx)
