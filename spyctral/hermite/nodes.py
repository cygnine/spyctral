# Nodes module for herfun

#__all__ = []

# Returns the scaling factor necessary so that half of the default N
# Gaussian nodes x satisfy |x|<=L
def scale_nodes(L,N,delta=0.5,mu=0.):

    from spyctral.hermite.quad import pgq
    from spyctral.common.scaling import scale_factor

    [x,w] = pgq(N,mu=0.)

    return scale_factor(L,x,scale=delta)
