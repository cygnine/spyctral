# !/usr/bin/env python
# 
# Routines for evaluation of polynomials

__all__ = ['eval_opoly', 'eval_opolyn']

import numpy as _np

# Evaluates the monic orthogonal polynomials at x defined by the recurrence constants
# a and b. It is assumed that a and b are long enough to evaluate the max(n)-th
# orthogonal polynomial. The d'th derivatives are evaluated. 
# The output is a length(d)-list containing length(x) by length(n) arrays
def eval_opoly(x,n,a,b,d=[0]) :

    # Preprocessing: unravel x and n arrays
    x = _np.array(x)
    n = _np.array(n)
    N = _np.max(n)
    D = _np.max(d)
    x = x.ravel()
    n = n.ravel()

    # Cast int d as list
    if type(d) != list :
        d = [d]
   
    # Allocation
    p = _np.zeros([x.size, N+2, D+1]);

    p[:,0,0] = 1.;
    
    if N==0 :
        return p[:,n,d]

    # Initial conditions for the recurrence relation, N=1
    p[:,1,0] = p[:,0,0]*(x-a[0])
    if D > 0 :
        p[:,1,1] = p[:,0,0]

    # Recurrence
    for q in range(N) :
        for qq in range(D+1) :
            p[:,q+2,qq] = (x-a[q+1])*p[:,q+1,qq] - b[q+1]*p[:,q,qq]
            if qq>0 :  # Correction for evaluation of derivatives
                p[:,q+2,qq] = p[:,q+2,qq] + qq*p[:,q+1,qq-1]

    if _np.size(d) > 1 :
        pr = []
        for q in d :
            pr.append(p[:,n,q])
    else :
        pr = p[:,n,d[0]]

    return pr

# Evaluates the orthonormal polynomials at x defined by the recurrence constants
# a and b. It is assumed that a and b are long enough to evaluate the max(n)-th
# orthogonal polynomial. The d'th derivatives are evaluated. 
# The output is a length(x) by length(n) by length(d) array
def eval_opolyn(x,n,a,b,d=[0]) :

    # Preprocessing: unravel x and n arrays
    x = _np.array(x)
    n = _np.array(n)
    N = _np.max(n)
    D = _np.max(d)
    x = x.ravel()
    n = n.ravel()

    # Cast int d as list
    if type(d) != list :
        d = [d]

    # Allocation
    p = _np.zeros([x.size, N+2, D+1])

    p[:,0,0] = 1./(b[0]**(1/2.))
    
    if N==0 :
        return p[:,n,d]

    # Initial conditions for the recurrence relation, N=1
    p[:,1,0] = 1./(b[1]**(1/2.))*p[:,0,0]*(x-a[0])
    if D > 0 :
        p[:,1,1] = p[:,0]

    for q in range(N) :
        for qq in range(D+1) :
            p[:,q+2,qq] = (x-a[q+1])*p[:,q+1,qq] - (b[q+1]**(1/2.))*p[:,q,qq]
            if qq>0 :  # Correction for evaluation of derivatives
                p[:,q+2,qq] = p[:,q+2,qq] + qq*p[:,q+1,qq-1]

            p[:,q+2,qq] = 1/(b[q+2]**(1/2.))*p[:,q+2,qq]


    if _np.size(d) > 1 :
        pr = []
        for q in d :
            pr.append(p[:,n,q])
    else :
        pr = p[:,n,d[0]]

    return pr
