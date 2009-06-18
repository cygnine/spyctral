# FFT module for wiener function package

__all__ = []

import numpy as _np
import scipy as _sp

# Uses the FFT to compute a modal expansion in the weighted generalized
# Wiener rational functions 
def fft_nodes_to_modes(f,s=1.,t=0.,shift=0.,scale=1.):

    from fourier.fft import fft as fft_Psi
    from wienerfun.eval import wx_sqrt
    from wienerfun.quad import genwiener_gquad as gq

    # This stuff can all be overhead
    N = f.size
    x = gq(N,s=1.,t=0.,shift=shift,scale=scale)[0]
    modes = f/wx_sqrt(x,s=s,t=t,shift=shift,scale=scale)

    temp = fft_Psi(modes,g=s-1.,d=t)*_np.sqrt(scale)
    temp[-s:] = 0
    temp[:s] = 0

    return temp

# Function that computes overhead stuff for FFT
def fft_nodes_to_modes_overhead(N,s=1.,t=0.,shift=0.,scale=1.):

    from numpy import sqrt
    from fourier.fft import fft_overhead
    from wienerfun.eval import wx_sqrt
    from wienerfun.quad import genwiener_gquad as gq

    overhead = fft_overhead(N,g=s-1.,d=t)
    x = gq(N,s=1.,t=0.,shift=shift,scale=scale)[0]
    premult = sqrt(scale)/wx_sqrt(x,s=s,t=t,shift=shift,scale=scale)

    return [premult,overhead]

# Function that does all the online fft computations
def fft_nodes_to_modes_online(f,overhead):

    from fourier.fft import fft_online

    temp = fft_online(f*overhead[0],overhead[1])

    # Is truncation of modes necessary?
    #s = overhead[1][1][0].shape[1]
    #temp[-s:] = 0
    #temp[:s] = 0

    return temp

# DOES NOT DO t>0 !!!!!
def fft_nodes_to_modes_galerkin(f,s=1.,t=0.,shift=0.,scale=1.):

    from fourier.fft import fft as fft_Psi
    from fourier.connection import int_connection as connect
    from numpy import sqrt,ones,abs,array
    from misc import pascal_row
    N = len(f)

    modes = fft_Psi(f)
    #eps = 1e-16
    #modes[abs(modes)<eps] = 0
    
    # Is there a better way to do this?
    for ss in range(int(s)):
        for count in range(N-1):
            modes[N-count-2] -= modes[N-count-1]
    modes *= 2**(s/2.)/(1j)**s

    # This way sucks
    #srow = array(pascal_row(s+1),dtype='complex128')
    #for count in range(int(s)-1):
    #    modes[N-count-2] -= sum(modes[N-count-1:]*srow[1:count+2])
    #for count in range(N-int(s)-1,-1,-1):
    #    modes[count] -= sum(modes[count+1:(count+s+1)]*srow[1:])

    modes = connect(modes,0,0,int(s-1),int(t))
    modes[:s] = 0
    modes[-s:] = 0
    return modes*sqrt(scale)

# Function that computes overhead stuff for FFT
def fft_galerkin_overhead(N,s=1.,t=0.,shift=0.,scale=1.):
    from numpy import sqrt
    from fourier.fft import fft_overhead
    from fourier.connection import int_connection_overhead

    foverhead = fft_overhead(N)
    premult = 2**(s/2.)/(1j)**s*sqrt(scale)
    coverhead = int_connection_overhead(N,0,0,int(s-1),int(t))

    return [premult,foverhead,coverhead,int(s)]

# Function that does all the online fft computations
def fft_galerkin_online(f,overhead):
    from fourier.fft import fft_online
    from fourier.connection import int_connection_online as connect
    from bidiag_invert import ones_bidiag_repeat

    N = f.size
    modes = fft_online(f*overhead[0],overhead[1])
    #
    # s = overhead[3]

    if overhead[3]>0:
        # Connect weight
        modes = ones_bidiag_repeat(modes,overhead[3])
        #for ss in range(int(overhead[3])):
        #    for count in range(N-1):
        #        modes[N-count-2] -= modes[N-count-1]

        modes = connect(modes,overhead[2])
        modes[:overhead[3]] = 0
        modes[-overhead[3]:] = 0

    return modes



# Uses the FFT to compute nodal evaluations given a modal expansion for
# the generalized Wiener rational functions
def fft_modes_to_nodes(F,s=1.,t=0.,shift=0.,scale=1.):

    from fourier.fft import ifft as ifft_Psi
    from wienerfun.eval import wx_sqrt
    from wienerfun.quad import genwiener_gquad as gq

    from fourier.fft import fft_overhead, ifft_online

    N = F.size

    #overhead = fft_overhead(N,g=s-1.,d=t)
    #fx = ifft_online(f,overhead)
    fx = ifft_Psi(F,g=s-1.,d=t)
    #fx = _np.zeros(N)
    
    x = gq(N,s=1.,t=0.,shift=shift,scale=scale)[0]
    fx *= wx_sqrt(x,s=s,t=t,shift=shift,scale=scale)/_np.sqrt(scale)
    #print "hi"

    return fx

# Function that does all the online stuff for ifft: the overhead is what
# is output from fft_modes_to_nodes_overhead: instead of multiplying by
# it, we'll just divide by it.
def fft_modes_to_nodes_online(f,overhead):

    from fourier.fft import ifft_online

    # overhead[1] = overhead from fourier.fft
    fx = ifft_online(f,overhead[1])

    # overhead[0] = 1/wx_sqrt
    return fx/overhead[0]
