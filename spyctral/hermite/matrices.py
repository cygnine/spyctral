# Module for performing operations involving Galerkin matrices for
# Hermite functions

# Returns the numarray vector corresponding to operating on the input
# vector `modes' with the L2 normalized Hermite function stiffness
# matrix. This is effectively a differentiation (since the mass matrix
# is the identity)
def stiff_apply(modes,mu=0.,shift=0.,scale=1.):

    from numpy import arange, zeros, sqrt

    N = modes.size

    dmodes = zeros(N)
    ns = arange(N)
    dmodes[:(N-1)] = sqrt(ns[1:]/2.)*modes[1:]
    dmodes[1:] -= sqrt((ns[:(N-1)]+1)/2.)*modes[:(N-1)]

    return dmodes/scale
