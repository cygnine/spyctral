#! /usr/bin/env python
# 
# Includes the Jacobi recurrence relation

import numpy as _np
import scipy.special as _nf

# Returns the first N Jacobi recurrence coefficients
def recurrence(N,alpha=-1/2.,beta=-1/2.,shift=0,scale=1) :

    alpha = float(alpha)
    beta = float(beta)
    a_s = (beta**2-alpha**2)*_np.ones([N,1])
    b_s = _np.zeros([N,1])

    a_s[0] = (beta-alpha)/(alpha+beta+2);
    b_s[0] = 2**(alpha+beta+1)*_nf.gamma(alpha+1)*_nf.gamma(beta+1)/_nf.gamma(alpha+beta+2)
    
    for q in range(2,N+1) :
        k = q-1
        a_s[k] = a_s[k]/((2*k+alpha+beta)*(2*k+alpha+beta+2))

        if k==1 :
            b_s[k] = 4*k*(k+alpha)*(k+beta)/((2*k+alpha+beta)**2*(2*k+alpha+beta+1))
        else :
            num = 4*k*(k+alpha)*(k+beta)*(k+alpha+beta)
            den =  (2*k+alpha+beta)**2*(2*k+alpha+beta+1)*(2*k+alpha+beta-1)
            b_s[k] = num/den

    return [a_s,b_s]
        
    # Still have recurrence_scaleshift to deal with
