# Module for common functions regarding spectral expansions

__all__ = []
import pdb

# Scales/shifts the input (numpy array) as indicated by the inputs to
# the unshifted/scaled version of the input for use in generating
# functions defined on standard intervals
# THIS ACTS ON MUTABLE INPUTS ONLY
def forward_scaleshift(x,scale=1.,shift=0.):
    x -= shift
    x /= scale

# Reverses the affine transformation performed by forward_scaleshift
# THIS ACTS ON MUTABLE INPUTS ONLY
def backward_scaleshift(x,scale=1.,shift=0.):
    x *= scale
    x += shift

# Returns the scaling factor necessary so that at least [scale] percent of the
# Gaussian nodes [x] satisfy | [x] |<= [L]. Mostly used for the infinite
# interval expansions.
def scale_factor(L,x,scale=0.5):

    from numpy import floor, ceil,abs,mean
    N = x.size
    scale = min(1.0,scale)
    
    Nfrac = ceil(N*scale)

    # Find scale interval [Lprev,Lnext] containing desired scale
    Lprev = 1e6
    Lnext = Lprev
    Nxs = sum(abs(x*Lnext)<=L)
    while Nxs<Nfrac:
        Lprev = Lnext
        Lnext /= 2
        Nxs = sum(abs(x*Lnext)<=L)

    # Find point in [Lprev,Lnext] at which number equals/exceeds Nfrac:
    Lmiddle = mean([Lprev,Lnext])
    Lsep = Lprev-Lmiddle
    Ltol = 1e-3
    while Lsep>Ltol:
        Nxs = sum(abs(x*Lmiddle)<=L)
        if Nxs<Nfrac:
            Lprev = Lmiddle
        else:
            Lnext = Lmiddle
        Lmiddle = mean([Lprev,Lnext])
        Lsep = Lprev-Lmiddle

    # Return a safe amount
    return Lnext
    #return Lmiddle/1.001

# Previous version of above function: hard coded for [scale] =0.5
def ScaleFactor(L,x):

    from numpy import floor, ceil
    N = x.size

    n = floor(N/2)
    centernode = ceil(N/2) - 1

    if not(bool(N%2)):
        if bool(n%2):
            loc = x[centernode+(n+1)/2]
        else:
            loc = 1/2.*(x[centernode+n/2] + x[centernode+n/2+1])
    else:
        if bool(n%2):
            loc = x[centernode+(n-1)/2]
        else:
            loc = 1/2.*(x[centernode+n/2] + x[centernode+n/2+1])

    return L/loc

# zero-pads to the left with zN zeros, to the right with zP zeros
def zero_pad(u,zN=0,zP=0):
    from numpy import hstack,zeros
    return hstack((zeros(zN),u,zeros(zP)))

# Removes zN modes from the left, zP modes from the right
def lowpass_filter(u,zN=0,zP=0):
    if zP==0:
        return u[zN:]
    else:
        return u[zN:-zP]

# Defines negative and positive # of zeros to pad/filter for N1<N2
# (Helper function for lowpass_filter and zero_pad)
def zp_define(N1,N2):
    if bool(N2%2):
        if bool(N1%2):
            zp_Nneg = (N2-N1)/2
            zp_Npos = (N2-N1)/2
        else:
            zp_Nneg = (N2-N1+1)/2
            zp_Npos = zp_Nneg - 1
    else:
        if bool(N1%2):
            zp_Nneg = (N2-N1)/2
            zp_Npos = zp_Nneg + 1
        else:
            zp_Nneg = (N2-N1)/2
            zp_Npos = zp_Nneg

    return [zp_Nneg,zp_Npos]

# Decimates a modal representation where the indexing is 0,1,...,N
def DecimateN(u,N,quiet=False):
    if (N>len(u)) and (not quiet):
        print "Cannot decimate a signal with a cutoff threshold above vector\
            length"
        return u
    else:
        return u[:N]

# Decimates a modal representation where the indexing is -N/2,..., 0,..., N/2
def DecimateK(u,N):
    if (N>len(u)) and (not quiet):
        print "Cannot decimate a signal with a cutoff threshold above vector\
            length"
        return u
    else:
        zs = zp_define(N,len(u))
        return lowpass_filter(u,zs[0],zs[1])

# Lowpass filters a modal representation where the indexing is 0,1,...,N
def LowPassN(u,N,quiet=False):
    from numpy import hstack,zeros
    if (N>len(u)) and (not quiet):
        print "Cannot low-pass filter a signal with a cutoff threshold above vector\
            length"
        return u
    else:
        return hstack((u[:N],zeros(len(u)-N)))

# Lowpass filters a modal representation where the indexing is 0,1,...,N
def LowPassK(u,N,quiet=False):
    if (N>len(u)) and (not quiet):
        print "Cannot low-pass filter a signal with a cutoff threshold above vector\
            length"
        return u
    else:
        zs = zp_define(N,len(u))
        return zero_pad(lowpass_filter(u,zs[0],zs[1]),zs[0],zs[1])
