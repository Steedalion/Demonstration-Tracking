#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 19:42:41 2019

@author: csteed
"""

import numpy as np;

def add_vectors_of_different_length(a,b):
    if len(a) < len(b):
        c = b.copy()
        c[:len(a)] += a
    else:
        c = a.copy()
        c[:len(b)] += b
    return c;

 def add_available(self, a:np.array, b:np.array):
        if len(a) < len(b):
            bigger = b;
            smaller = a;       
        else:
            bigger = a;
            smaller = b;
        
        c = bigger.copy()
        dimx, dimy = smaller.shape;
        c[:dimx,:dimy] += smaller
        return c;