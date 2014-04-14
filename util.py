#!/usr/bin/env python2.7
"""
Various utility methods
"""

def tuple_add(t1, t2):
    return tuple( t1[i] + t2[i] for i in xrange(len(t1)) )

