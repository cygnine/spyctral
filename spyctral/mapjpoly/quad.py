#!/usr/bin/env python
# Quadrature module for the Jacobi functions mapped to the real line

#__all__ = ['gq',
#           'pgq']

#def quad(N,s=1.,t=1.,scale=1.,shift=0.):
#    return pgq(N,s=s,t=t,scale=scale,shift=shift)

def gq(N,s=1.,t=1.,scale=1.,shift=0.):
    from spyctral.common.maps import physical_scaleshift as pss
    from spyctral.mapjpoly.maps import st_to_ab, r_to_x
    from spyctral.jacobi.quad import gq

    [alpha,beta]=st_to_ab(s,t)
    [r,w] = gq(N,alpha=alpha,beta=beta)
    x = r_to_x(r)

    #x *= scale 
    #x += shift
    pss(x,scale=scale,shift=shift)
    return [x,w]

def pgq(N,s=1.,t=1.,scale=1.,shift=0.):
    from spyctral.mapjpoly.maps import weight
    
    [x,w] = gq(N,s=s,t=t,scale=scale,shift=shift)

    # Weight accordingly
    w /= weight(x,s=s,t=t,scale=scale,shift=shift)
    return [x,w]
