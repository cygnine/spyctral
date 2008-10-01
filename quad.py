# !/usr/bin/env python
# 
# Routines for finding orthogonal polynomial quadrature

__all__ = ['opoly_gq']

from numpy import array, sqrt, diag
import numpy as _np
from scipy import sparse
from numpy.linalg import eigh

def opoly_gq(a,b,N) :

    a = array(a);
    b = array(b);

    assert (a.ndim==1)&(b.ndim==1), "Recurrence inputs a and b must be 1-D arrays"

    Ntmp = max(a.size,b.size)
    if N>Ntmp : 
        print "Downgrading N from %d to %d" % (N,Ntmp)
        N = Ntmp;

    # Define Jacobi matrix
    J = diag(a[:N]) + diag(sqrt(b[1:N]),1) + diag(sqrt(b[1:N]),-1)
    # Fortran can't do sparse stuff built-in:
#    J = sparse.lil_matrix([N,N])
#    J.setdiag(a[:N])
#    J.setdiag(sqrt(b[1:N]),1)
#    J.setdiag(sqrt(b[1:N]),-1)

    [x,d] = eigh(J)

    w = b[0]*(d[0,:]**2)

    return [x,w]
