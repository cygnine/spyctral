# Nodes module for wienerfun

__all__ = []

# Returns the scaling factor necessary so that half of the default N
# Gaussian nodes x satisfy |x|<=L
def scale_nodes(L,N,s=1.,t=0.):

    from quad import genwiener_gquad as gq
    from spectral_common import scale_factor

    [x,w] = gq(N,s=s,t=t)

    return scale_factor(L,x)
