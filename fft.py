# Module to implement built-in numpy fft for computing fourier modes
# 
# 20090302 -- acn

__all__ = []

# Uses numpy fft to obtain modes. Only works on 1D arrays
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
