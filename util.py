#!/usr/bin/env python2.7
"""
Various utility methods
"""
import operator

def tuple_add(t1, t2):
    """Elementwise addition of two tuples"""
    return tuple( t1[i] + t2[i] for i in xrange(len(t1)) )

def nonzeros(lst):
    """Return non-zero indices of a list"""
    return (i for i in xrange(len(lst)) if lst[i] > 0)

def first(iterable, default=None, key=None):
    if key is None:
        for el in iterable:
            return el
    else:
        for el in iterable:
            return el
    return default

def prod(iterable):
    return reduce(operator.mul, iterable, 1)

def to_syms(R, *monoms):
    """
    Get the symbols of an ideal I
    """
    return [prod(R.symbols[i]**j 
                for (i, j) in enumerate(monom)) 
                    for monom in monoms]

