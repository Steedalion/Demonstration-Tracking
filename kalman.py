#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 13:12:39 2019

@author: csteed
"""
import numpy as np;
import matplotlib.pyplot as plot;

T = 1/30;

F = np.array([[1, T],[0, 1]]);np.zeros(np.shape(F));
F = np.block([
        [F, np.zeros(np.shape(F))],
        [np.zeros(np.shape(F)),F]
        ]);

Gamma = np.array([[pow(T,2)/2],[T]]);
Gamma = np.block([
        [Gamma, np.zeros(np.shape(Gamma))],
        [np.zeros(np.shape(Gamma)),Gamma]
        ]);

H = np.array([[1,0,0,0],[0,0,1,0]]);

z = np.zeros([2,50])
tmax = len(z[1]);
x_true = np.zeros([4,tmax]);
nx = np.size(x_true,0);
x_estimate = np.zeros(np.shape(x_true));

x_true[:,0] = np.array([1,1,0,1.5])
pk = np.zeros((tmax,nx,nx))
pk[0] = np.eye(np.size(x_true,0));
R = 0.002;
Q = np.eye(2)*0.000001;

t = np.linspace(0,tmax*T,tmax)

for i in range(1,tmax):
    x_true[:,i] = F.dot(x_true[:,i-1]) +Gamma.dot(np.random.normal(0,np.sqrt(Q)).dot(np.ones([2,1]))).T    
    z[:,i] = H.dot(x_true[:,i]) +np.random.normal(0,np.sqrt(R))
    
    #time update
    x_priori = np.dot(F,x_estimate[:,i-1]);
    P_negative = F.dot(pk[i-1]).dot(F.T) + Gamma.dot(Q).dot(Gamma.T)
    
    #y=     
    K = P_negative.dot(H.T).dot(np.linalg.inv(H.dot(P_negative).dot(H.T) + R))
    #K = pk.dot(H.T)/((H.dot(pk).dot(H.T) + R))
    pk[i] = (np.eye(np.size(P_negative,0)) - K.dot(H))*(P_negative)
    x_estimate[:,i] = x_priori + K.dot(z[:,i] - H.dot(x_priori))
    
    
    #measurement update
    
    #print(x_priori)
    
plot.figure(1)
plot.subplot(1,2,1)
plot.title("Position")
plot.plot(z[0,:],z[1,:],'r.'
          ,x_estimate[0,:],x_estimate[2,:],'b')
plot.subplot(1,2,2)
plot.title("Velocity")
plot.plot(t,x_true[2,:],'r',t,x_estimate[2,:],'b')

