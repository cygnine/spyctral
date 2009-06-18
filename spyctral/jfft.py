# !/usr/bin/env python
# Module to load constants necessary for using the FFT to perform Jacobi
# polynomial transformations. 

import opoly1.jacobi as jacobi
from scipy import sparse
import numpy as _np

__all__ = ['gamma','umat','vmat','rmatrix']

# Returns the first N Jacobi gamma constants. These constants satisfy the
# recurrence relation (1-r)*P_n^(a,b) = g(1)*P_(n+1)^(a-1,b) + g(2)*P_n^(a-1,b). 
# Therefore the inputs must satisfy a>0, b>-1.
# NB: these are also called 'mu' in the paper
def gamma(N,a=1/2.,b=-1/2.) :

    a = float(a)
    b = float(b)
    g = _np.zeros([N,2])
    ns = _np.array(range(N))
    tol = 1e-12

    temp = 2*ns+a+b

    if abs(a+b)<tol :
        g[0,0] = (2.*a/(a+b+1.))**0.5
        g[1:,0] = (2.*(ns[1:]+a)*(ns[1:]+a+b)/(temp[1:]*(temp[1:]+1.)))**0.5
    else :
        g[:,0] = (2.*(ns+a)*(ns+a+b)/(temp*(temp+1.)))**0.5

    g[:,1] = (2.*(ns+1.)*(ns+b+1.)/((temp+1.)*(temp+2.)))**0.5

    return g

# Defines the U matrix, a square N x N bidiagonal matrix that is used as a
# building block for the full transformation matrix R. The matrix U transforms
# the (alpha,beta) modes to the (alpha+1,beta) modes. Returns a sparse csc
# matrix U.
def umat(N,alpha,beta) :

    gs = gamma(N,alpha,beta)
    return sparse.spdiags([gs[:,0], -gs[:,1]],[0,1],N,N)
    #return sparse.spdiags([gs[:,0], -gs[:,1]],[0,1],N,N).todense()

# Defines the V matrix, a square N x N bidiagonal matrix that is used as a
# building block for the full transformation matrix R. The matrix U transforms
# the (alpha,beta) modes to the (alpha,beta+1) modes. Returns a sparse csc
# matrix U.
def vmat(N,alpha,beta) :

    gs = gamma(N,beta,alpha)
    return sparse.spdiags([gs[:,0], gs[:,1]],[0,1],N,N)
    #return sparse.spdiags([gs[:,0], gs[:,1]],[0,1],N,N).todense()

# Performs the operation V*R where V is the bidiagonal sparse matrix defined by
# vmat. This avoids construction of the actual matrix V
def _vmat_direct_mult(R,N,alpha,beta) :

    gs = gamma(N,beta,alpha)

    for q in range(N-1) :
        R[q,:] = (gs[q,0]*R[q,:] + gs[q,1]*R[q+1,:]).todense()

    R[N-1,:] = (gs[N-1,0]*R[N-1,:]).todense()

# Performs the operation U*R where U is the bidiagonal sparse matrix defined by
# vmat. This avoids construction of the actual matrix U
def _umat_direct_mult(R,N,alpha,beta) :

    gs = gamma(N,alpha,beta)

    for q in range(N-1) :
        R[q,:] = (gs[q,0]*R[q,:] - gs[q,1]*R[q+1,:]).todense()

    R[N-1,:] = (gs[N-1,0]*R[N-1,:]).todense()

# Defines the R matrix, a square N x N bidiagonal matrix that is used as a
# building block for the full transformation matrix R. Returns a sparse csc
# matrix R. 
# In this function, R is constructed via sparse matrix multiplication. This is
# slow.
def rmatrix(N,alpha,beta,A,B) :

    A = int(round(A))
    B = int(round(B))

    R = sparse.lil_eye([N,N]).tocsc()
    #R = sparse.lil_eye([N,N]).todense()

    for q in range(A) :
        #_umat_direct_mult(R,N,alpha+q+1,beta)
        #R = umat(N,alpha+q+1,beta)*R
        R = umat(N,alpha+q+1,beta).matmat(R)
        #R = _np.dot(umat(N,alpha+q+1,beta),R)

    for q in range(B) :
        #_vmat_direct_mult(R,N,alpha+A,beta+q+1)
        #R = vmat(N,alpha+A,beta+q+1)*R
        R = vmat(N,alpha+A,beta+q+1).matmat(R)
        #R = _np.dot(vmat(N,alpha+A,beta+q+1),R)

    return R.tocsc()

# Defines the R matrix, a square N x N bidiagonal matrix that is used as a
# building block for the full transformation matrix R.  Returns a sparse csc
# matrix R.
# In this function, R is constructed via direct arithmetic operations on the
# recurrence constants. This is fast.
def rmatrix_direct(N,alpha,beta,A,B) :

    A = int(round(A))
    B = int(round(B))
    # Each column of Rs: diagonal of R. The column index is the diagonal offset
    Rs = _np.zeros([N,A+B+1])
    Rs[:,0] = 1;

    for q in range(A) :
        gs = gamma(N,alpha+q+1,beta)
        Rs[:N-1,1:q+2] = Rs[:N-1,1:q+2]*gs[:N-1,0:1] - Rs[1:N,0:q+1]*gs[:N-1,1:2]
        Rs[:N,0:1] *= gs[:N,0:1]

    for q in range(A,A+B) :
        gs = gamma(N,beta+q-A+1,alpha+A)
        Rs[:N-1,1:q+2] = Rs[:N-1,1:q+2]*gs[:N-1,0:1] + Rs[1:N,0:q+1]*gs[:N-1,1:2]
        Rs[:N,0:1] *= gs[:N,0:1]

    return sparse.spdiags(Rs.T,range(A+B+1),N,N)

# Computes the elements of the R matrix, a square N x N bidiagonal matrix that is used as a
# building block for fast Jacobi transformations. Returns an N x (A+B) array.
# In this function, the entries are constructed via direct arithmetic operations on the
# recurrence constants. This is fast.
def rmatrix_entries(N,alpha,beta,A,B) :

    A = int(round(A))
    B = int(round(B))
    # Each column of Rs: diagonal of R. The column index is the diagonal offset
    Rs = _np.zeros([N,A+B+1])
    Rs[:,0] = 1;

    for q in range(A) :
        gs = gamma(N,alpha+q+1,beta)
        Rs[:N-1,1:q+2] = Rs[:N-1,1:q+2]*gs[:N-1,0:1] - Rs[1:N,0:q+1]*gs[:N-1,1:2]
        Rs[:N,0:1] *= gs[:N,0:1]

    for q in range(A,A+B) :
        gs = gamma(N,beta+q-A+1,alpha+A)
        Rs[:N-1,1:q+2] = Rs[:N-1,1:q+2]*gs[:N-1,0:1] + Rs[1:N,0:q+1]*gs[:N-1,1:2]
        Rs[:N,0:1] *= gs[:N,0:1]

    if ((A==B)&(alpha==beta)):
        for q in range(A):
            Rs[:,2*q+1] = 0

    return Rs

# Uses the result of rmatrix_entries to compute the modal transformation
def rmatrix_entries_apply(u,Rs):

    N = u.size
    v = Rs[:,0]*u.copy()
    AB = Rs.shape[1]-1

    #for q in range(N) :
    #    indmin = 0
    #    indmax = min(q+2+AB,N+1)-(q+1)
    #    v[q] = _np.sum(Rs[q,indmin:indmax]*u[q+indmin:q+indmax])

    for q in range(AB):
        tmp = N-q-1
        v[:tmp] += Rs[:tmp,q+1]*u[(q+1):]

    return v

# Uses the function rmatrix_entries to derive the Rs entries, and inverts the
# matrix R via back-substitution. Assumes the input u is a modal expansion in
# Jacobi polynomials of order (alpha+A,beta+B), and reverts it to the output v,
# a modal expansion in Jacobi polynomials of order (alpha,beta)
def rmatrix_invert(u,alpha,beta,A,B) :

    v = u.copy()
    N = u.size
    Rs = rmatrix_entries(N,alpha,beta,A,B)

    for q in range(N-1,-1,-1) :
        indmin = 1
        indmax = min(q+2+A+B,N+1)-(q+1)
        v[q] -= _np.sum(v[q+indmin:q+indmax]*Rs[q,indmin:indmax])
        v[q] /= Rs[q,0]

    return v

# Is an `online' version of rmatrix_invert: the R matrix entries are
# given as input
def rmatrix_entries_invert(u,Rs):
    from bidiag_invert import uppertri_diagonal_invert

    v = u.copy()
    N = u.size
    AB = Rs.shape[1]-1

    #for q in range(N-1,-1,-1) :
    #    indmin = 1
    #    indmax = min(q+2+AB,N+1)-(q+1)
    #    v[q] -= _np.sum(v[q+indmin:q+indmax]*Rs[q,indmin:indmax])
    #    v[q] /= Rs[q,0]

    v = uppertri_diagonal_invert(u,Rs)

    return v

# Applies the matrix R via the entries rmatrix_entries. Transforms u into the
# output v
def rmatrix_apply(u,alpha,beta,A,B):

    N = u.size
    v = u.copy()
    Rs = rmatrix_entries(N,alpha,beta,A,B)

    for q in range(N) :
        indmin = 0
        indmax = min(q+2+A+B,N+1)-(q+1)
        v[q] = _np.sum(Rs[q,indmin:indmax]*u[q+indmin:q+indmax])

    return v

# Applies the matrix R via sequential operations: hopefully this reduces the
# conditioning problem of the full R matrix?
def rmatrix_apply_seq(u,alpha,beta,A,B):

    N = u.size
    v = u.copy()

    A = int(round(A))
    B = int(round(B))
    C = min(A,B)
    ns = _np.arange(N)
    tol = 1e-8

    for c in range(C):
        if _np.abs(alpha+beta+1)<tol:
            ab = alpha+beta
            nab = ns[1:]+alpha+beta
            n2ab = 2*ns[1:]+alpha+beta

            gsc = _np.zeros(N)
            gsc[0] = 2/(ab+2)*_np.sqrt((alpha+1)*(beta+1)*(ab+2)/
                    (ab+3))
            gsc[1:] = 2/(n2ab+2)*_np.sqrt((ns[1:]+alpha+1)*
                    (ns[1:]+beta+1)*(nab+1)*(nab+2)/
                    ((n2ab+1)*(n2ab+3)))
            ab = alpha+beta
            nab = ns+alpha+beta
            n2ab = 2*ns+alpha+beta
        else: 
            ab = alpha+beta
            nab = ns+alpha+beta
            n2ab = 2*ns+alpha+beta
            gsc = 2/(n2ab+2)*_np.sqrt((ns+alpha+1)*
                    (ns+beta+1)*(nab+1)*(nab+2)/
                    ((n2ab+1)*(n2ab+3)))

        v = gsc*u

        gsc = 2*_np.sqrt((ns+1)*(ns+2)*(ns+beta+1)*(ns+alpha+2)/
                ((n2ab+2)*(n2ab+3)*(n2ab+4)*(n2ab+5)))

        v[:(N-2)] -= gsc[:(N-2)]*u[2:]

        #v = gsa[:,0]*gsb[:,0]*u
        #v[:(N-2)] -= gsa[:(N-2),1]*gsb[1:(N-1),1]*u[2:]
        #v = rmatrix_apply(v,alpha,beta,1,1)
        u = v.copy()
        
        alpha += 1
        beta += 1

    if A>C:
        for a in range(A-C):
            gs = gamma(N,alpha+1,beta)
            v = gs[:,0]*u
            v[:(N-1)] -= gs[:(N-1),1]*u[1:]
            u = v.copy()
            #u = rmatrix_apply(u,alpha,beta,1,0)
            alpha += 1
    elif B>C:
        for b in range(B-C):
            gs = gamma(N,beta+1,alpha)
            v = gs[:,0]*u
            v[:(N-1)] += gs[:(N-1),1]*u[1:]
            u = v.copy()
            #u = rmatrix_apply(u,alpha,beta,0,1)
            beta += 1

    return u

# Performs the Cheybshev FFT forward transform
def chebfft(f,scale=1.):
    from numpy import array, sqrt, exp, hstack
    from numpy.fft import rfft as fft
    from scipy import pi
    N = f.size

    shift = -array(range(N))*(1+1/(2.*N))
    shift = sqrt(2*pi)*exp(1j*pi*shift.T)/(2.*N)
    shift[0] /= sqrt(2)
    shift *= sqrt(scale)

    F = fft(hstack((f,f[::-1])),axis=0)[0:N]
    return (F*shift).real

# Performs the inverse Chebyshev transform with an FFT
def chebifft(F,scale=1.):
    from numpy import array, sqrt, exp, hstack
    from numpy.fft import irfft as ifft
    from scipy import pi
    N = F.size

    shift = -array(range(N))*(1+1/(2.*N))
    shift = sqrt(2*pi)*exp(1j*pi*shift.T)/(2.*N)
    shift[0] /= sqrt(2)
    shift *= sqrt(scale)

    f = F/shift
    f = ifft(hstack((f,array([0]))),axis=0,n=2*N)
    return f[:N].real

# This uses the FFT to compute modal expansions of the nodal evaluations at the
# Chebyshev nodes. This method essentially tranlates the Chebyshev interpolant
# to the Gegenbauer modes. Assumes fx are nodal values at the Chebyshev-Gauss
# nodes
def jacfft(fx,A,B,scale=1.) :

    from numpy.fft import rfft as fft
    from numpy import sqrt,exp
    from scipy import pi
    
    """
    Deprecated stuff (see chebfft)
    N = fx.size

    # Preliminary shifting to get fft to line up with Cheb transform
    shift = -_np.array(range(N))*(1+1/(2.*N))
    shift = sqrt(2*pi)*exp(1j*pi*shift.T)/(2.*N)
    shift[0] /= sqrt(2)

    # Perform fft with shift/scale
    F = _np.zeros(fx.shape)
    F = fft(_np.hstack((fx,fx[::-1])),axis=0)[0:N]
    F = (F*shift).real
    """

    F = chebfft(fx,scale=scale)
    
    # Apply R matrix to get modes for (-1/2+A,-1/2+B) expansion
    #F = rmatrix_apply(F,-1/2.,-1/2.,A,B)
    F = rmatrix_apply_seq(F,-1/2.,-1/2.,A,B)

    return F

# This uses the FFT to compute modal expansions of the nodal evaluations at the
# Chebyshev nodes. This method essentially tranlates the Chebyshev interpolant
# to the Jacobi modes. Assumes fx are nodal values at the Chebyshev-Gauss
# nodes.
def jacifft(F,A,B,scale=1.):

    from numpy.fft import irfft as ifft
    from numpy import sqrt,exp
    from scipy import pi
    N = F.size

    # Connection problem back to Chebyshev modes
    F = rmatrix_invert(F,-1/2.,-1/2.,A,B)

    # Use inverse chebyshev transform
    return chebifft(F,scale=scale)


# Computes all necessary overhead for doing a Chebyshev fft
def chebfft_overhead(N,scale=1.):
    from numpy import array, sqrt, exp
    from scipy import pi

    shift = -array(range(N))*(1+1/(2.*N))
    shift = sqrt(2*pi)*exp(1j*pi*shift.T)/(2.*N)
    shift[0] /= sqrt(2)
    shift *= sqrt(scale)
    return shift

# Performs the online chebyshev fft
def chebfft_online(f,overhead):
    from numpy.fft import rfft as fft
    from numpy import hstack

    N = f.size
    F = fft(hstack((f,f[::-1])),axis=0)[0:N]
    return (F*overhead).real

# Performs the ifft using the fft_overhead:
def chebifft_online(F,overhead):
    from numpy.fft import irfft as ifft
    from numpy import hstack,array

    N = F.size
    f = F/overhead
    f = ifft(hstack((f,array([0]))),axis=0,n=2*N)
    return f[:N].real

# Computes overhead for general Jacobi polynomial transform
def jacfft_overhead(N,A,B,scale=1.):

    cheb_overhead = chebfft_overhead(N,scale=scale)
    connection = rmatrix_entries(N,-1/2.,-1/2.,A,B)

    return [cheb_overhead,connection]

# Computes the online Jacobi fft:
# overhead[0] = chebfft_overhead
# overhead[1] = rmatrix_entries
def jacfft_online(f,overhead):

    # Chebyshev FFT
    F = chebfft_online(f,overhead[0])

    # Connection problem
    return rmatrix_entries_apply(F,overhead[1])

# Computes the online inverse Jacobi fft:
# overhead[0] = chebfft_overhead
# overhead[1] = rmatrix_entries
def jacifft_online(F,overhead):

    # Connection problem (inverse)
    f = rmatrix_entries_invert(F,overhead[1])

    # Chebysehv FFT
    return chebifft_online(f,overhead[0])
