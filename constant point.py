#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 17:05:07 2019

@author: csteed
"""
import numpy as np
import matplotlib.pyplot as plot
F = np.eye(3)

x0 = np.hstack([1,0,1])
t_final = 100;
x_store = np.zeros([3,t_final]).T;
x_store[:,] = x0;
for i in range(1,100):
    x_store[i] = F.dot(x_store[i-1]) 
   
plot.plot(x_store)