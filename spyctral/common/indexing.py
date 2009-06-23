#!/usr/bin/env python
"""
* File Name : indexing.py

* Creation Date : 2009-06-16

* Created By : Akil Narayan

* Last Modified : Tue 16 Jun 2009 05:40:32 PM EDT

* Purpose : Indexing package for common module of pyspec module. Contains basic
*   operations for indexing of spectral methods. 
"""

def whole_range(N):
    """Returns indices for whole-number type indexing"""
    from numpy import arange
    return arange(int(N))

def integer_range(N):
    """Returns indices for integer-number type indexing"""
    from numpy import arange

    N = int(N)
    if bool(N%2):
        N = (N-1)/2
        ks = arange(-N,N+1)
    else:
        N = N/2
        ks = arange(-N,N)

    return ks

def whole_etas(N):
    """Returns eta indicators for modal indices: these are the indicators for
    input into spectral filter methods"""

    Ns = whole_range(N)
    return Ns/float(N-1)

def integer_etas(N):
    """Returns eta indicators for modal indices: these are the indicators for
    input into spectral filter methods"""

    ks = integer_range(N)
    if bool(N%2):
        N = (N-1)/2
    else:
        N = N/2
    
    return ks/float(N)
