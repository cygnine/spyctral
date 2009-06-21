#!/usr/bin/env python
#
# Module for evaluation of mapped Jacobi polynomials
#
# Parameters: s = decay at -Inf, t = decay at +Inf
# 2*s = 2*a + 3
# 2*t = 2*b + 3
# For (s,t)<-->(a,b) map, see maps.py

#__all__ = ['jfunn',
#           'wjfunn']

# Evaluates the unweighted mapped Jacobi functions, orthogonal under the
# weight function maps.wjacobiw
# jfunn
def jacobi_function(x,ns,s=1.,t=1.,scale=1.,shift=0.):
    from spyctral.jacobi.eval import jpoly
    from spyctral.mapjpoly.maps import x_to_r, st_to_ab
    [alpha,beta] = st_to_ab(s,t)

    return jpoly(x_to_r(x,scale=scale,shift=shift),
            ns,alpha=alpha,beta=beta)

# Evaluates the derivatives of the unweighted mapped Jacobi functions, orthogonal under the
# weight function maps.wjacobiw
# djfunn
def djacobi_function(x,ns,s=1.,t=1.,scale=1.,shift=0.):
    from spyctral.jacobi.eval import djpoly
    from spyctral.mapjpoly.maps import x_to_r, st_to_ab, dr_dx
    [alpha,beta] = st_to_ab(s,t)

    jac = dr_dx(x,scale=scale,shift=shift)

    r = x_to_r(x,scale=scale,shift=shift)
    ps = djpoly(r,ns,alpha=alpha,beta=beta)

    return (ps.T*jac).T

# Evaluates the weighted mapped Jacobi functions, orthogonal under unit
# weight on the real line.
# wjfunn
def weighted_jacobi_function(x,ns,s=1.,t=1.,scale=1.,shift=0.):
    from spyctral.mapjpoly.maps import sqrt_weight

    w = sqrt_weight(x,s=s,t=t,scale=scale,shift=shift)

    return (jacobi_function(x,ns,s=s,t=t,scale=scale,shift=shift).T*w).T

# Evaluates the derivatives of the weighted mapped Jacobi functions,
# orthogonal under unit weight on the real line.
# dwjfunn
def dweighted_jacobi_function(x,ns,s=1.,t=1.,scale=1.,shift=0.):
    from spyctral.mapjpoly.maps import sqrt_weight, dsqrt_weight

    w = sqrt_weight(x,s=s,t=t,scale=scale,shift=shift)
    dw = dsqrt_weight(x,s=s,t=t,scale=scale,shift=shift)

    PB = jacobi_function(x,ns,s=s,t=t,scale=scale,shift=shift)
    dPB = djacobi_function(x,ns,s=s,t=t,scale=scale,shift=shift)

    # Yay product rule
    return (PB.T*dw).T + (dPB.T*w).T
