#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 13:12:39 2019

@author: csteed
"""
import numpy as np;
import matplotlib.pyplot as plot;

T = 1/30;

F = np.array([[1, T],[0, 1]]);
Gamma = np.array([[pow(T,2)/2],[T]]).T

H = np.array([1,0]);

position_true = np.zeros([2,50]);
position_true[:,0] = np.array([1,1])

pk = np.array(1.0);
R = np.array(0.0001);
Q = np.array(0.00001);

position_estimate = np.zeros(np.shape(position_true));
tmax = len(position_true[1]);
t = np.linspace(0,tmax*T,tmax)
for i in range(1,tmax):
    position_true[:,i] = F.dot(position_true[:,i-1]) +np.random.normal(0,np.sqrt(Q))
    
    z = position_true[0,i] +np.random.normal(0,np.sqrt(R))
    
    #time update
    x_priori = np.dot(F,position_estimate[:,i-1]);
    pk = F.dot(pk).dot(F.T) + Q
    
    #y=     
    #K = pk.dot(H.T).dot(np.linalg.inv(H.dot(pk).dot(H.T) + R))
    K = pk.dot(H.T)/((H.dot(pk).dot(H.T) + R))
    pk = (np.eye(np.size(K.dot(H))) - K.dot(H))*(pk)
    position_estimate[:,i] = x_priori + K.dot(z - H.dot(x_priori))
    
    
    #measurement update
    
    #print(x_priori)
    
plot.figure(1)
plot.title("Position")
plot.plot(position_true[0,:],'r'
          ,position_estimate[0,:],'b')
plot.figure(2)
plot.title("Velocity")
plot.plot(t,position_true[1,:],'r',t,position_estimate[1,:],'b')