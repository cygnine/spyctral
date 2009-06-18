# Nodes module for Fourier

__all__ = []

# Returns the scaling factor necessary so that half of the default N
# Gaussian nodes x satisfy |x|<=L
def scale_nodes(L,N,delta=0.5,g=0.,d=0.):

    from quad import genfourier_gquad as gq
    from spyctral.common import scale_factor

    [x,w] = gq(N,g=g,d=d)

    return scale_factor(L,x,scale=delta)
