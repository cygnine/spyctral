#!/usr/bin/env python
"""
* File Name : maps.py

* Creation Date : 2009-06-17

* Created By : Akil Narayan

* Last Modified : Wed 17 Jun 2009 04:58:46 PM EDT

* Purpose : Generic mapping routines for spectral domains
"""

def standard_scaleshift(x,shift=0.,scale=1):
    """
    Shifts the input x defined on a physical domain scale*I+shift to that
    defined on the standard domain I.  Assumes x is mutable.
    """

    x -= shift
    x /= scale

def physical_scaleshift(x,shift=0.,scale=1.):
    """
    Shifts the input x defined on the standard domain I to that defined on the
    physical domain scale*I+shift.  Assumes x is mutable.  
    """

    x *= scale
    x += shift
