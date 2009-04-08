# Nodes module for Fourier

__all__ = []

# Returns the scaling factor necessary so that half of the default N
# Gaussian nodes x satisfy |x|<=L
def scale_nodes(L,N,g=0.,d=0.):

    from quad import genfourier_gquad as gq
    from spectral_common import scale_factor

    [x,w] = gq(N,g=g,d=d)

    return scale_factor(L,x)
