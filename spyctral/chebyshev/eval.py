#!/usr/bin/env python
"""
* File Name : eval.py

* Creation Date : 2009-06-17

* Created By : Akil Narayan

* Last Modified : Wed 17 Jun 2009 03:43:12 PM EDT

* Purpose :
"""

def cpoly(x,n,d=0,normalization='normal',scale=1., shift=0.):
    from pyspec.jacobi.eval import jpoly

    return jpoly(x,n,d=d,alpha=-1/2.,beta=-1/2.,normalization=normalization, \
            scale=scale, shift=shift)
