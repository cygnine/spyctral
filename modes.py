# !/usr/bin/env python
#
# Module for doing modal-nodal transformations for cardinal Whittaker
# splines

# Given a function f, output N modes. If N is even, bias is to the
# left (negative).
def get_N_modes_from_function(f,N,Q='sinc',shift=0.,scale=1.):

    from quad import N_to_ks

    ks = N_to_ks(N)
    return get_ks_modes_from_function(f,ks,Q=Q,shift=shift,scale=scale)

# Given a function f, output the modes associated with the given
# indices ks.
def get_ks_modes_from_function(f,ks,s=1.,t=0.,Q='wf',shift=0.,scale=1.):

    from eval import genwienerw
    from numpy import dot

    if Q=='wf':
        from quad import genwienerw_pgquad as pgq
        [x,w] = pgq(ks.size,s=s,t=t,shift=shift,scale=scale)
    elif any(type(Q)==i for i in [int, float]):
        from quad import genwienerw_pgquad as pgq
        [x,w] = pgq(Q,shift=shift,scale=scale)
    else:
        [x,w] = Q

    ps = genwienerw(x,ks,s=s,t=t,shift=shift,scale=scale)

    return dot(ps.T*w,f(x))

# Given the modes associated with indices ks, return a function that
# evaluates the interpolant.
def modal_expansion_ks(modes,ks,s=1.,t=0.,shift=0.,scale=.1):

    from eval import genwienerw as gw
    from numpy import sum

    return lambda(y): sum(modes*gw(y,ks,s=s,t=t,shift=shift,scale=scale),axis=1)

# Given a collection of modes, assumes the standard default distribution 
# of modes and returns a function that evaluates the interpolant at
# nodes.
def modal_expansion(modes,s=1.,t=0.,shift=0.,scale=1.):

    from quad import N_to_ks
    from numpy import array

    N = array(modes).size
    ks = N_to_ks(N)
    return modal_expansion_ks(modes,ks,s=s,t=t,shift=shift,scale=scale)

# Given modes associated with indices ks, return the interpolant
# evaluated at the locations y
def modal_expansion_ks_at_y(modes,ks,y,s=1.,t=0.,shift=0.,scale=1.):

    return modal_expansion_ks(modes,ks,s=s,t=t,shift=shift,scale=scale)(y)

# Given a collection of modes, assumes the standard default distribution
# of modes and returns the value of the interpolant at the locations y.
def modal_expansion_at_y(modes,y,s=1.,t=0.,shift=0.,scale=1.):

    return modal_expansion(modes,s=s,t=t,shift=0.,scale=1.)(y)

# Given a function f, evaluate the ks-indexed interpolant function
def f_to_modal_expansion_ks(f,ks,s=1.,t=0.,Q='wf',shift=0.,scale=1.):

    modes = get_ks_modes_from_function(f,ks,s=s,t=t,Q=Q,shift=shift,scale=scale)
    return modal_expansion_ks(modes,ks,s=s,t=t,shift=shift,scale=scale)

# Given a function f, evaluate the ks-indexed interpolant at the
# values y.
def f_to_modal_expansion_ks_at_y(f,ks,y,s=1.,t=0.,Q='wf',shift=0.,scale=1.):
    
    return modal_expansion_ks(f,ks,s=s,t=t,Q=Q,shift=shift,scale=scale)(y)

# Given a function f, return a default standard N-mode interpolant
def f_to_modal_expansion(f,N,s=1.,t=0.,Q='wf',shift=0.,scale=1.):

    from quad import N_to_ks

    ks = N_to_ks(N)
    return f_to_modal_expansion_ks(f,ks,s=s,t=t,Q=Q,shift=shift,scale=scale)

# Given a function f, return its default standard N-point interpolant at
# the locations y
def f_to_modal_expansion_at_y(f,N,y,s=1.,t=0.,Q='wf',shift=0.,scale=1.):

    return f_to_modal_expansion(f,N,s=s,t=t,Q=Q,shift=shift,scale=scale)(y)
