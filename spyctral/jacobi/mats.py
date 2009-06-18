def stiff_apply(F,alpha=-1/2.,beta=-1/2.,scale=1.):
    """
    Applies the modal stiffness matrix to the coefficients of the L2 normalized
    polynomials. Makes use of the recurrence constant zetan in addition to the
    sparse representation of the connection coefficients Is an O(N) operation
    """

    from jfft import rmatrix_invert
    from numpy import arange,hstack, array

    N = F.size
    # Input F is of class (alpha,beta). Take the derivative by promoting
    # basis functions to (alpha+1,beta+1) using zetan:
    zetas = zetan(arange(N),alpha=alpha,beta=beta)/scale

    # Now demote the (alpha+1,beta+1) coefficients back down to
    # (alpha,beta)
    filler = array([0.])
    return hstack((rmatrix_invert(zetas[1:]*F[1:],alpha,beta,1,1),filler))

# Calculates overhead required for applying the modal stiffness matrix
def stiff_overhead(N,alpha=-1/2.,beta=-1/2.,scale=1.):
    from jfft import rmatrix_entries
    from numpy import arange

    zetas = zetan(arange(N),alpha=alpha,beta=beta)/scale
    Rs = rmatrix_entries(N-1,alpha,beta,1,1)
    return [zetas[1:],Rs]

# Performs the online application of the stiffness matrix given the
# overhead from stiff_overhead as input
def stiff_online(F,overhead):
    from jfft import rmatrix_entries_invert
    from numpy import array,hstack

    filler = array([0.])
    return hstack((rmatrix_entries_invert(overhead[0]*F[1:],overhead[1]),\
            filler))
