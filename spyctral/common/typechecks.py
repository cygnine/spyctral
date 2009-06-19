#!/usr/bin/env python

# Common module in pyspec package

def TestListType(x,name=''):
    from numpy import ndarray
    assert isinstance(x,(list,ndarray)), "Input " + name + " must be a list or \
        numpy array"
