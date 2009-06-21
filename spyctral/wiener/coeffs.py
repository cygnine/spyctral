# Coefficient module for the general Wiener basis function package

import numpy as _np

#__all__ = ['genwienerw_stiff']

# Returns constants for representing -s/(x-i)*phi_k^(s) in terms of the phik.
# Each row contains six columns: kvee, -kvee, k, -k, kwedge, -kwedge
def weighted_wiener_lemma1_entries(ks,s=1.,t=0.):

    from numpy import sign as sgn

    ks = _np.array(ks,dtype='int')
    ks = ks.ravel()

    entries = _np.zeros([ks.size,6],dtype='complex128')
    al = -1/2.
    be = s-3/2.
    kcount = 0
    i = 1j

    for k in ks:
        n = _np.abs(k)-1
        ka = _np.abs(k)
        tempab = 2*n+al+be

        temp = -i*s/2/(tempab+2)*_np.sqrt((n+al+1)*(n+be+1)/((tempab+1)*(tempab+3)))
        entries[kcount,0] = temp*\
           (_np.sqrt(n+al+be+1)*sgn(k)+_np.sqrt(n))*(sgn(k)*_np.sqrt(n+1) + _np.sqrt(n+al+be+2))
        entries[kcount,1] = temp*\
           (_np.sqrt(n+al+be+1)*sgn(k)-_np.sqrt(n))*(sgn(k)*_np.sqrt(n+1) + _np.sqrt(n+al+be+2))
        entries[kcount,2] = -i*s/2*((be-al)*(al+be+1)/((tempab+2)*(tempab+4))+1)
        entries[kcount,3] = -i*s/2*(al-be)/((tempab+2)*(tempab+4))*(1+sgn(k)*2*_np.sqrt((n+1)*(n+al+be+2)))
        temp = -i*s/(2*(tempab+4))*_np.sqrt((n+al+2)*(n+be+2)/((tempab+3)*(tempab+5)))
        entries[kcount,4] = temp*\
           (sgn(k)*_np.sqrt(n+2)-_np.sqrt(n+al+be+3))*(sgn(k)*_np.sqrt(n+al+be+2)- _np.sqrt(n+1))
        entries[kcount,5] = temp*\
           (sgn(k)*_np.sqrt(n+2)+_np.sqrt(n+al+be+3))*(sgn(k)*_np.sqrt(n+al+be+2)- _np.sqrt(n+1))

        kcount+=1

    return entries


# Returns constants for the unweighted Wiener function stiffness matrix
# Each row contains six columns: kvee, -kvee, k, -k, kwedge, -kwedge
def wiener_stiff_entries(ks,s=1.,t=0.):

    import spyctral.opoly1d.jacobi as jac
    from numpy import sign as sgn

    ks = _np.array(ks,dtype='int')
    ks = ks.ravel()

    entries = _np.zeros([ks.size,6],dtype='complex128')
    al = -1/2.
    be = s-3/2.
    kcount = 0

    for k in ks:
        n = _np.abs(k)-1
        ka = _np.abs(k)
        tempab = 2*n+al+be

        #zeta = jac.zetan(ka,al,be)
        #factor = (n+al+be+2)*_np.sqrt(2*(n+al+1)*(n+al+be+2)/((tempab+2)*(tempab+3)))
        #factorp1 = (n+1)*_np.sqrt(2*(n+1)*(n+be+2)/((tempab+3)*(tempab+4)))
        #fs = _np.array([factor,factorp1])
        #gammasn = jac.gamman(n,be+1,al)
        #gammasnp1 = jac.gamman(n+1,be+1,al)
        #[a_s,b_s] = jac.recurrence_ns([ka-1,ka],al+1,be+1)

        # kvee, -kvee, k, -k, kwedge, -kwedge
        #entries[kcount,0] = (fs[0]*gammasn[0]+zeta*_np.sqrt(b_s[0])) 
        entries[kcount,0] = (n+al+be+2)*_np.sqrt((n+al+1)*(n+be+1)/((tempab+1)*(tempab+2)**2*(tempab+3)))* \
                            (_np.sqrt((n+al+be+1)*(n+al+be+2)) + _np.sqrt(n*(n+1)))

        #entries[kcount,1] = (fs[0]*gammasn[0]-zeta*_np.sqrt(b_s[0]))
        entries[kcount,1] = (n+al+be+2)*_np.sqrt((n+al+1)*(n+be+1)/((tempab+1)*(tempab+2)**2*(tempab+3)))* \
                            (_np.sqrt((n+al+be+2)*(n+al+be+1)) - _np.sqrt(n*(n+1)))
        #entries[kcount,2] = (fs[0]*gammasn[1]+fs[1]*gammasnp1[0]+zeta*(a_s[0]+1))
        entries[kcount,2] = _np.sqrt((n+1)*(n+al+be+2))
        #entries[kcount,3] = (fs[0]*gammasn[1]+fs[1]*gammasnp1[0]-zeta*(a_s[0]+1))
        entries[kcount,3] = (al+be+2)*(al-be)/((tempab+2)*(tempab+4))*\
                            _np.sqrt((n+1)*(n+al+be+2))
        #entries[kcount,4] = (fs[1]*gammasnp1[1]+zeta*_np.sqrt(b_s[1]))
        entries[kcount,4] = (n+1)*_np.sqrt((n+be+2)*(n+al+2)/((tempab+3)*(tempab+4)**2*(tempab+5)))* \
                            (_np.sqrt((n+1)*(n+2)) + _np.sqrt((n+al+be+2)*(n+al+be+3)))
        #entries[kcount,5] = (fs[1]*gammasnp1[1]-zeta*_np.sqrt(b_s[1]))
        entries[kcount,5] = (n+1)*_np.sqrt((n+be+2)*(n+al+2)/((tempab+3)*(tempab+4)**2*(tempab+5)))* \
                            (_np.sqrt((n+1)*(n+2)) - _np.sqrt((n+al+be+2)*(n+al+be+3)))

        entries[kcount,:] *= 1j*sgn(k)
        kcount += 1

    return entries

# Returns constants for the weighted Wiener function stiffness matrix
# Each row contains six columns: kvee, -kvee, k, -k, kwedge, -kwedge
def wieghted_wiener_stiff_entries(ks,s=1.,t=0.,scale=1.):

    from numpy import sign as sgn
    from numpy import sqrt

    ks = _np.array(ks,dtype='int')
    ks = ks.ravel()

    entries = _np.zeros([ks.size,6],dtype='complex128')
    al = -1/2.
    be = s-3/2.
    kcount = 0
    i = 1j

    for k in ks:
        n = _np.abs(k)-1
        ka = _np.abs(k)
        tempab = 2*n+al+be

        # Special expression if k=0
        if k==0:
            entries[kcount,2] = -i*(s-1/2.)
            entries[kcount,4] = -i/2*sqrt(s-1/2.)*(1-sqrt(s))/sqrt(1+s)
            entries[kcount,5] = -i/2*sqrt(s-1/2.)*(1+sqrt(s))/sqrt(1+s)
        else:
            # Special expression if |k|=1
            if n==0:
                entries[kcount,0] = i/_np.sqrt(2)*_np.sqrt((al+1)*(be+1)/((al+be+3)))*(sgn(k)*\
                _np.sqrt((al+be+2)) - 1)
            else:
                entries[kcount,0] = i/2*_np.sqrt((n+al+1)*(n+be+1)/((tempab+1)*(tempab+3)))*\
                    (sgn(k)*(_np.sqrt((n+al+be+1)*(n+al+be+2)) + _np.sqrt(n*(n+1))) -\
                    (al+be+2)/(tempab+2)*(_np.sqrt((n+1)*(n+al+be+1)) + _np.sqrt(n*(n+al+be+2))))
                entries[kcount,1] = i/2*_np.sqrt((n+al+1)*(n+be+1)/((tempab+1)*(tempab+3)))*\
                    (sgn(k)*(_np.sqrt((n+al+be+1)*(n+al+be+2)) - _np.sqrt(n*(n+1))) -\
                    (al+be+2)/(tempab+2)*(_np.sqrt((n+1)*(n+al+be+1)) - _np.sqrt(n*(n+al+be+2))))

            entries[kcount,2] = i*sgn(k)*_np.sqrt((n+1)*(n+al+be+2)) - \
                i*s*(be-al)*(al+be+1)/(2*(tempab+2)*(tempab+4)) - i*s/2

            entries[kcount,3] = -s/2*i*(al-be)/((tempab+2)*(tempab+4))

            entries[kcount,4] = i/2*_np.sqrt((n+be+2)*(n+al+2)/((tempab+3)*(tempab+5))) *\
                (-s/(tempab+4)*(_np.sqrt((n+2)*(n+al+be+2)) + _np.sqrt((n+1)*(n+al+be+3))) +\
                sgn(k)*(_np.sqrt((n+al+be+2)*(n+al+be+3)) + _np.sqrt((n+1)*(n+2))))

            entries[kcount,5] = i/2*_np.sqrt((n+be+2)*(n+al+2)/((tempab+3)*(tempab+5))) *\
                (-s/(tempab+4)*(_np.sqrt((n+2)*(n+al+be+2)) - _np.sqrt((n+1)*(n+al+be+3))) +\
                sgn(k)*(-_np.sqrt((n+al+be+2)*(n+al+be+3)) + _np.sqrt((n+1)*(n+2))))

        kcount += 1

    return entries/scale

# Function that calculates overhead for applying the 
# stiffness matrix for the weighted Wiener rational functions. The
# coefficients should be the output of weighted_wiener_stiff_entries.
def apply_stiff_entries_overhead(N,s=1.,t=0.,scale=1.):

    print """This function is not supported due to the efficiency of the CSR
    matvec routine in scipy. genwienerw_stiff() returns a CSR
    representation of the stiffness matrix"""

    from wienerfun.quad import N_to_ks
    ks = N_to_ks(N)

    # Generate the coefficients
    entries = weighted_wiener_stiff_entries(ks,s=s,t=t,scale=scale)

    # Generate necessary maps to appl
    if entries.size[0] != N:
        print """Error: inputs f and coefficients must have the same
               number of rows"""

    ######## INCOMPLETE: does built-in CSR multiplication do better?
    # For N = 1000, .matvec() routine does things ~ 40 times faster than
    # numpy.dot, so I'm abandoning this routine for now
