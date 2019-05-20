#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 09:09:20 2019

@author: csteed
"""
import numpy as np;
n = 1000000;
sigma = 1
mu = 0

s = np.random.normal(mu, sigma, n)


import matplotlib.pyplot as plt

count, bins, ignored = plt.hist(s, 30, density=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
               np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
         linewidth=2, color='r')
plt.show()