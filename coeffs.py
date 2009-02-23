# Coefficient module for Fourier package

import numpy as _np

__all__ = []

# Recurrence coefficients for cosine recurrence:
# cos*Psi_k = sum(c[i]*Psi_i), 
# i = [-kwedge,-k,-kvee,kvee,k,kwedge]
def cos_recurrence(ks,g,d):

    from numpy import zeros, abs,sqrt
    from opoly1.jacobi import recurrence_ns as jrec
    from opoly1.jacobi import recurrence as jprec

    coeffs = zeros([ks.size,6])
    al = d-1/2.
    be = g-1/2.

    ks0 = (ks==0)
    ks1 = (abs(ks)==1)
    ks2 = ~(ks0 | ks1)

    n = abs(ks)
    N = _np.max(n)
    [a,b] = jprec(N+2,al,be)
    b = sqrt(b)
    [a1,b1] = jprec(N+2,al+1,be+1)
    b1 = sqrt(b1)

    # 0 = -kwedge
    # 1 = -k
    # 2 = -kvee
    # 3 = kvee
    # 4 = k
    # 5 = kwedge
    if any(ks0):
        #b = sqrt(jrec(n[ks0]+1,al,be)[1])
        #b /= sqrt(2)
        #a = jrec(n[ks0],al,be)[0]
        coeffs[ks0,0] = b[1]/sqrt(2)
        coeffs[ks0,5] = b[1]/sqrt(2)
        coeffs[ks0,4] = a[0]
    if any(ks1):
        #[a1,b1] = jrec([0,1,2],al+1,be+1)
        #[a,b] = jrec([0,1,2],al,be)
        #b1 = sqrt(b1)
        #b = sqrt(b)
        coeffs[ks1,0] = 1/2.*(b[2] - b1[1])
        coeffs[ks1,1] = 1/2.*(a[1] - a1[0])
        coeffs[ks1,3] = sqrt(2)/2*b[1]
        coeffs[ks1,4] = 1/2.*(a[1] + a1[0])
        coeffs[ks1,5] = 1/2.*(b[2] + b1[1])
    if any(ks2):
        #[a1,b1] = jrec(n[ks2],al+1,be+1)
        #[a,b] = jrec(n[ks2],al,be)
        #b1 = sqrt(b1)
        #b = sqrt(b)
        np1 = n[ks2]+1
        nm1 = n[ks2]-1
        n = n[ks2]
        coeffs[ks2,0] = 1/2.*(b[n+1] - b1[n])
        coeffs[ks2,1] = 1/2.*(a[n] - a1[n-1])
        coeffs[ks2,2] = 1/2.*(b[n] - b1[n-1])
        coeffs[ks2,3] = 1/2.*(b[n] + b1[n-1])
        coeffs[ks2,4] = 1/2.*(a[n] + a1[n-1])
        coeffs[ks2,5] = 1/2.*(b[n+1] + b1[n])

    return coeffs.squeeze()

# Recurrence coefficients for sine recurrence:
# i*sin*Psi_k = sum(c[i]*Psi_i), 
# i = [-kwedge,-k,-kvee,kvee,k,kwedge]
def sin_recurrence(ks,g,d):

    from numpy import zeros, abs,sqrt
    from numpy import sign as sgn
    from opoly1.jacobi import etan, epsilonn 

    coeffs = zeros([ks.size,6])
    al = d-1/2.
    be = g-1/2.

    ks0 = (ks==0)
    ks1 = (abs(ks)==1)
    ks2 = ~(ks0 | ks1)

    n = abs(ks)
    N = _np.max(n)
    eta = etan(range(N+2),al,be)
    eps = epsilonn(range(N+2),al+1,be+1)

    # 0 = -kwedge
    # 1 = -k
    # 2 = -kvee
    # 3 = kvee
    # 4 = k
    # 5 = kwedge
    if any(ks0):
        coeffs[ks0,0] = -1/sqrt(2)*eta[0,2]
        coeffs[ks0,5] = 1/sqrt(2)*eta[0,2]
    if any(ks1):
        coeffs[ks1,0] = sgn(ks[ks1])/2.*(-eta[1,2]-eps[0,2])
        coeffs[ks1,1] = sgn(ks[ks1])/2.*(-eta[1,1]-eps[0,1])
        coeffs[ks1,3] = sgn(ks[ks1])/-sqrt(2)*eps[0,0]
        coeffs[ks1,4] = sgn(ks[ks1])/2.*(eta[1,1]-eps[0,1])
        coeffs[ks1,5] = sgn(ks[ks1])/2.*(eta[1,2]-eps[0,2])
    if any(ks2):
        n = n[ks2]
        coeffs[ks2,0] = sgn(ks[ks2])/2.*(-eta[n,2]-eps[n-1,2])
        coeffs[ks2,1] = sgn(ks[ks2])/2.*(-eta[n,1]-eps[n-1,1])
        coeffs[ks2,2] = sgn(ks[ks2])/2.*(-eta[n,0]-eps[n-1,0])
        coeffs[ks2,3] = sgn(ks[ks2])/2.*(eta[n,0]-eps[n-1,0])
        coeffs[ks2,4] = sgn(ks[ks2])/2.*(eta[n,1]-eps[n-1,1])
        coeffs[ks2,5] = sgn(ks[ks2])/2.*(eta[n,2]-eps[n-1,2])

    return coeffs.squeeze()

# Recurrence coefficients for exp recurrence:
# exp(i*theta)*Psi_k = sum(c[i]*Psi_i), 
# i = [-(k+1),-k,-(k-1),(k-1),k,(k+1)]
def exp_recurrence(ks,g,d):
    
    from numpy import zeros, abs,sqrt
    from numpy import sign as sgn
    coeffs = zeros([ks.size,6])

    al = d - 1/2.
    be = g - 1/2.
    n = abs(ks)
    tempab = 2*n+al+be

    ks0 = (n == 0)
    ks1 = (n == 1)
    # The kwedge expressions are (mostly) the same for any n
    tempab = 2*n+al+be
    kwedge = sqrt((n+al+1)*(n+be+1))/((tempab+2)*sqrt((tempab+1)*(tempab+3)))
    kwedge *= (sqrt(n+al+be+1) + sqrt(n))
    coeffs[:,0] = kwedge*(sqrt(n+1)-sqrt(n+al+be+2))
    coeffs[:,5] = kwedge*(sqrt(n+1)+sqrt(n+al+be+2))
    coeffs[ks0,[0,5]] *= sqrt(2)

    # Now we can't do n=0 without special cases:
    n = n[~ks0]

    tempab = 2*n+al+be
    den = tempab*(tempab+2)
    coeffs[~ks0,1] = (be-al)*(-1+2*sqrt(n*(n+al+be+1)))/den
    coeffs[~ks0,4] = (be-al)*(al+be+1)/den

    kvee = sqrt((n+al)*(n+be))/(tempab*sqrt((tempab-1)*(tempab+1)))
    kvee *= (sqrt(n) - sqrt(n+al+be+1))
    coeffs[~ks0,2] = kvee*(sqrt(n+al+be)+sqrt(n-1))
    coeffs[~ks0,3] = kvee*(sqrt(n+al+be)-sqrt(n-1))

    # Deal with ks0, ks1:
    coeffs[ks1,2] = 0.
    coeffs[ks1,3] *= sqrt(2)

    coeffs[ks0,4] = (be-al)/(al+be+2)

    return coeffs.squeeze()

# Recurrence coefficients for exp recurrence:
# exp(i*theta)*Psi_k = sum(c[i]*Psi_i), 
# i = [-kwedge,-k,-kvee,kvee,k,kwedge]
def exp_add_recurrence(ks,g,d):
    
    return sin_recurrence(ks,g,d)+cos_recurrence(ks,g,d)

# Six-term recurrence coeffs:
# D_n*Psi_{n+1} = (A_n*e^(itheta) - B_n)*Psi_n + (A_-n*e^(-itheta) - B_-n)*Psi_-n
# + C_n Psi_{n-1} + C_-n Psi_{-(n-1)}
def recurrence(ns,g,d):

    from numpy import sqrt

    if type(ns)==list:
        ns = _np.array(ns)
    else:
        ns = _np.array([ns]).ravel()

    al = d-1/2.
    be = g-1/2.
    N = _np.max(ns)

    D = _np.zeros(N)
    Ap = D.copy()
    Am = D.copy()
    Bp = D.copy()
    Bm = D.copy()
    Cp = D.copy()
    Cm = D.copy()

    n0 = (ns==0)
    n1 = (ns==1)
    nn = ~ (n0 | n1)

    ta = sqrt(ns+al+be+2)
    tb = sqrt(ns+1)
    Ap = ta + tb
    Am = ta - tb

    n2ab = 2*ns+al+be
    na1 = ns+al+1
    nb1 = ns+be+1
    nab = ns+al+be
    D = 4*sqrt(na1*nb1*(ns+1)*(nab+2)/((n2ab+1)*(n2ab+3)))/(n2ab+2)*\
         (sqrt(nab+1)+sqrt(ns))

    ta = (be-al)/(n2ab*(n2ab+2))
    Bp = ta*\
            ((al+be+1)*(sqrt(nab+2)+sqrt(ns+1)) + (2*sqrt(ns*(nab+1))-1)*\
            (sqrt(nab+2)-sqrt(ns+1)))
    Bm = ta*\
            ((al+be+1)*(sqrt(nab+2)-sqrt(ns+1)) + (2*sqrt(ns*(nab+1))-1)*\
            (sqrt(nab+2)+sqrt(ns+1)))


    ta = 2*(al+be+1)/n2ab*sqrt((ns+al)*(ns+be)\
            /((n2ab-1)*(n2ab+1)))/(sqrt(ns)+sqrt(nab+1))
    Cp = ta*(sqrt((nab+1)**2-1)-sqrt(ns**2-1))
    Cm = ta*(sqrt((nab+1)**2-1)+sqrt(ns**2-1))

    # Correct for n=0
    D[n0] *= 2
    Bm[n0] = 0
    Bp[n0] = 2*sqrt(2)*(be-al)/sqrt(al+be+2)
    Cp[n0] = 0
    Cm[n0] = 0
    Ap[n0] *= sqrt(2)
    Am[n0] *= sqrt(2)

    # Correct for n=1
    Cm[n1] /= sqrt(2)
    Cp[n1] /= sqrt(2)

    return [Ap,Am,Bp,Bm,Cp,Cm,D]
