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

    from jfft import rmatrix_apply_seq as jconnection
    G = int(G)
    D = int(D)
    if (G+D)>0:

        N = modes.shape[0]

        modes *= 1/2.
        cmodes = c_collapse(modes,N)
        smodes = s_collapse(modes,N)

        cmodes = jconnection(cmodes,d-1/2.,g-1/2.,D,G)
        smodes = jconnection(smodes,d+1/2.,g+1/2.,D,G)

        return c_expand(cmodes,N) + s_expand(smodes,N)
    else:
        return modes

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
