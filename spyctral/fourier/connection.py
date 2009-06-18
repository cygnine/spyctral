# Module for implementing the connection coefficients for the Szego-Fourier
# functions
#
# 20090119 -- acn

import numpy as _np

# Implements the connection coefficient relation, taking the modes from
# Szego-Fourier class (g,d) to Szego-Fourier class (g+G,d+D), and casts G and D
# as integers so that the connection relation is sparse. 
# The convention assumed for an even number of modes is that they are centered
# around the zero frequency, and that there is one more negative mode than
# positive modes.
# RIGHT NOW ONLY WORKS FOR G,D>0
def int_connection(modes,g,d,G,D):

    from spyctral.jfft import rmatrix_apply_seq as jconnection
    G = int(G)
    D = int(D)
    if (G+D)>0:

        N = modes.shape[0]

        [cmodes,smodes] = sc_collapse(modes,N)

        cmodes = jconnection(cmodes,d-1/2.,g-1/2.,D,G)
        smodes = jconnection(smodes,d+1/2.,g+1/2.,D,G)

        return sc_expand(cmodes,smodes,N)
    else:
        return modes.copy()

# Reverses the integer connection performed by int_connection
# The input modes is a vector of (g+G,d+D) modes, and we wish to demote
# them back to (g,d) modes.
def int_connection_backward(modes,g,d,G,D):

    from spyctral.jfft import rmatrix_invert as jconnection_inv
    G = int(G)
    D = int(D)
    if (G+D)>0:

        N = modes.size

        [cmodes,smodes] = sc_collapse(modes,N)

        cmodes = jconnection_inv(cmodes,d-1/2.,g-1/2.,D,G)
        smodes = jconnection_inv(smodes,d+1/2.,g+1/2.,D,G)
        
        return sc_expand(cmodes,smodes,N)
    else:
        return modes.copy()

def sc_collapse(modes,N):
    from numpy import sqrt

    tempN = N/2
    tempN2 = tempN - ((N+1)%2)

    cmodes = modes[:tempN+1][::-1].copy()
    smodes = -cmodes[1:].copy()
    
    cmodes[1:tempN2+1] += modes[tempN+1:]
    smodes[:tempN2] += modes[tempN+1:]
    cmodes[0] *= sqrt(2)

    return [cmodes*1/2.,smodes*1/2.]

# Helper function: combines roles of c_expand, s_expand
def sc_expand(cmodes,smodes,N):
    from numpy import sqrt,zeros
    Neven = (N+1)%2
    n = N/2
    factor = N - Neven
    cmodes[0] *= sqrt(2)
    
    # 'positive' modes
    pmodes = cmodes.copy()
    pmodes[1:] += smodes
    # Turn smodes into 'negative' modes
    smodes = (cmodes[1:] - smodes)[::-1]

    # Put it all together
    modes = zeros(2*n+1,dtype='complex128')
    modes[:n],modes[n:] = smodes,pmodes
    if bool(Neven):
        return modes[:-1]
    else:
        return modes

# Returns the necessary matrices for performing the int_connection; use
# as an overhead to plug into int_connection_online.
def int_connection_overhead(N,g,d,G,D):
    from spyctral.jfft import rmatrix_entries as jconnection_entries
    from numpy import zeros,ceil
    G = int(G)
    D = int(D)

    cconnect = jconnection_entries((N+2)/2,d-1/2.,g-1/2.,D,G)
    sconnect = jconnection_entries(N/2,d+1/2.,g+1/2.,D,G)
    cmodes = zeros(ceil((N+1.)/2),dtype='complex128')
    smodes = zeros(N/2,dtype='complex128')
    NEven = not(N%2)
    # location of mode 0
    Nmiddle = N/2

    return [cconnect,sconnect,cmodes,smodes,NEven,Nmiddle]
    #return [cconnect,sconnect]

# Applies the connection matrices returned by int_connection_overhead to
# perform the fft.
def int_connection_online(modes,matrices):
    from spyctral.jfft import rmatrix_entries_apply as jconnection_apply
    from numpy import sqrt,zeros,hstack,flipud
    # ONLY USE THE FOLLOWING LINE FOR C/FORTRAN TIMINGS
    #from jfft_helpers import rmatrixapply as jconnection_apply
    #from jfft_helpers import rmatrixapplyfast as jconnection_apply
    #from FourierHelpers import rmatrixapplyfourier as jconnection_apply

    if matrices[0].shape[1]>1:
        N = modes.shape[0]
        # How to read the code below:
        # cmodes = matrices[2]
        # smodes = matrices[3]
        # NEven = matrices[4]
        Nmiddle = matrices[5]

        ### sc_collapse ###
        # Ensures modes is odd-length
        if matrices[4]:
            modes = hstack((modes,0.))

        matrices[3] = flipud(modes[:Nmiddle])
        matrices[2][1:] = 1/2.*(modes[(Nmiddle+1):] + matrices[3])
        matrices[2][0] = sqrt(2)/2*modes[Nmiddle]
        matrices[3] = 1/2.*(modes[(Nmiddle+1):] - matrices[3])

        ### connection ###
        #matrices[2] = jconnection_apply(matrices[2],matrices[0])
        #matrices[3] = jconnection_apply(matrices[3],matrices[1])
        [matrices[2],matrices[3]] = jconnection_apply(matrices[2],matrices[0],\
                                    matrices[3],matrices[1])

        #### sc_expand ###
        modes[Nmiddle] = matrices[2][0]*sqrt(2)
        modes[(Nmiddle+1):] = matrices[2][1:]+matrices[3]
        modes[Nmiddle-1::-1] = (matrices[2][1:]-matrices[3])
        if matrices[4]:
            return modes[:-1]
        else:
            return modes

        # Using utility functions:
        #[cmodes,smodes] = sc_collapse(modes,N)
        #cmodes = jconnection_apply(cmodes,matrices[0])
        #smodes = jconnection_apply(smodes,matrices[1])
        #return sc_expand(cmodes,smodes,N)

    else: 
        return modes.copy()
        #return modes

def int_connection_backward_online(modes,matrices):
    from spyctral.jfft import rmatrix_entries_invert as jconnection_apply

    if matrices[0].shape[1]>1:
        N = modes.shape[0]

        [cmodes,smodes] = sc_collapse(modes,N)

        cmodes = jconnection_apply(cmodes,matrices[0])
        smodes = jconnection_apply(smodes,matrices[1])

        return sc_expand(cmodes,smodes,N)
    else: 
        return modes.copy()



###################### DEPRECATED ######################

# Helper function: collapses to even modes
def c_collapse(modes,N):
    tempN = (N+2)/2
    temp = modes[:tempN][::-1].copy()
    factor = temp.size - (1+(-1)**N)/2
    temp[1:factor] += modes[tempN:]
    temp[0] *= _np.sqrt(2)
    return temp

# Helper function: collapses to odd modes
def s_collapse(modes,N):
    tempN = N/2
    temp = -modes[:tempN][::-1].copy()
    factor = temp.size - (1+(-1)**N)/2
    temp[:factor] += modes[(tempN+1):]
    return temp

# Helper function: expands even modes
def c_expand(modes,N):
    factor = modes.size - (1+(-1)**N)/2
    return _np.hstack((modes[1:][::-1],_np.array([modes[0]*_np.sqrt(2)]),modes[1:factor]))

# Helper function: expands odd modes
def s_expand(modes,N):
    factor = modes.size - (1+(-1)**N)/2
    return _np.hstack((-modes[::-1],_np.array([0.]),modes[:factor]))
