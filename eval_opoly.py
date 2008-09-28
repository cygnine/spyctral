#! /usr/bin/env python
# Evaluates orthogonal polynomials given the recurrence constants
#
# 20080927 -- acn

import numpy as _np

# Evaluates the monic orthogonal polynomials at x defined by the recurrence constants
# a and b. It is assumed that a and b are long enough to evaluate the max(n)-th
# orthogonal polynomial. 
# The output is a length(x) by length(n) array
def eval_opoly(x,n,a,b,d=0) :

    # Preprocessing: unravel x and n arrays
    x = _np.array(x)
    n = _np.array(n)
    N = _np.max(n)
    x = x.ravel()
    n = n.ravel()

    p = _np.zeros([x.size, N+2, d+1]);

    p[:,0,0] = 1.;
    
    if N==0 :
        return p[:,n,d]

    p[:,1] = p[:,0]*(x-a[0])

    for q in range(N) :
        p[:,q+2] = (x-a[q+1])*p[:,q+1] - b[q+1]*p[:,q]

    return p[:,n]
