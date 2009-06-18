# !/usr/bin/env python
# 
# Routines for evaluation of polynomials

__all__ = ['eval_opoly', 'eval_normalized_opoly']

import numpy as _np

# Evaluates the monic orthogonal polynomials at x defined by the recurrence constants
# a and b. It is assumed that a and b are long enough to evaluate the max(n)-th
# orthogonal polynomial. The d'th derivatives are evaluated. 
# The output is a length(d)-list containing length(x) by length(n) arrays
def eval_opoly(x,n,a,b,d=[0]):
    from numpy import array, max, zeros
    from scipy import factorial
    from pyspec.common.typechecks import TestListType

    map(TestListType,[x,n,a,b], ['x','n','a','b'])

    # Preprocessing: unravel x and n arrays
    x = array(x).ravel()
    n = array(n).ravel()
    N = max(n)
    D = max(d)

    # Cast int d as list
    if type(d) != list :
        d = [d]
   
    # Allocation
    p0 = zeros([x.size, N+1])
    p1 = zeros([x.size, N+1])
    p = zeros([x.size, len(n), len(d)]);

    if N==0 :
        p[:,n==0,d==0] = 1.
        return p.squeeze()

    # Recurrence
    Dcount = 0
    for qq in range(D+1):
        # start recurrence
        p1[:,:qq] = 0.
        p1[:,qq] = factorial(qq)
        p1[:,qq+1] = (x-a[qq])*p1[:,qq] + qq*p0[:,qq]
    
        for q in range(2+qq,N+1):
            p1[:,q] = (x-a[q-1])*p1[:,q-1] - b[q-1]*p1[:,q-2] + qq*p0[:,q-1]
        # Assignment
        if qq in d:
            p[:,:,Dcount] = p1[:,n]
            Dcount += 1
        p0 = p1.copy()

    return p.squeeze()

# Evaluates the orthonormal polynomials at x defined by the recurrence constants
# a and b. It is assumed that a and b are long enough to evaluate the max(n)-th
# orthogonal polynomial. The d'th derivatives are evaluated. 
# The output is a length(x) by length(n) by length(d) array
def eval_normalized_opoly(x,n,a,b,d=[0]):

    from numpy import array, max, zeros, sqrt, prod
    from scipy import factorial
    from pyspec.common.typechecks import TestListType

    map(TestListType,[x,n,a,b], ['x','n','a','b'])

    # Preprocessing: unravel x and n arrays
    x = array(x).ravel()
    n = array(n).ravel()
    N = max(n)
    D = max(d)

    # Cast int d as list
    if type(d) != list :
        d = [d]
   
    # Allocation
    p0 = zeros([x.size, N+1])
    p1 = zeros([x.size, N+1])
    p = zeros([x.size, len(n), len(d)]);

    # Preprocessing on recurrence coefficients
    bsq = sqrt(b)

    if N==0 :
        p[:,n==0,d==0] = 1./bsq[0]
        return p.squeeze()

    # Recurrence
    Dcount = 0
    for qq in range(D+1):
        # start recurrence
        p1[:,:qq] = 0.
        p1[:,qq] = factorial(qq)/prod(bsq[:(qq+1)])
        p1[:,qq+1] = 1/bsq[qq+1]*((x-a[qq])*p1[:,qq] + qq*p0[:,qq])
    
        for q in range(2+qq,N+1):
            p1[:,q] = 1/bsq[q]*((x-a[q-1])*p1[:,q-1] - \
                                bsq[q-1]*p1[:,q-2] + \
                                qq*p0[:,q-1])

        # Assignment
        if qq in d:
            p[:,:,Dcount] = p1[:,n]
            Dcount += 1
        p0 = p1.copy()

    return p.squeeze()
