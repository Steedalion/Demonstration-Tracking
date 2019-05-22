#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 13:12:39 2019

@author: csteed
"""
import numpy as np;
import matplotlib.pyplot as plot;

def backPass(F, P_neg, P_pos, x_neg, x_pos):
    x_new = np.zeros(np.shape(x_pos));
    x_new = np.array(x_pos)
    P_new = np.array(P_pos)
    #x_new = x_pos
    P_new = np.zeros(np.shape(P_neg));
    t = np.size(x_pos,1);
    
    for i in range(t-1,0,-1):
        print(i)
        L = P_pos[i-1].dot(F.T).dot(np.linalg.inv(P_neg[i]));
        x_new[:,i-1] = x_pos[:,i-1] + L.dot(x_new[:,i]-x_neg[:,i]);
        P_new[i-1] = P_pos[i-1] + L.dot(P_new[i]-P_neg[i]).dot(L.T);
    return (P_new,x_new);

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
x_priori = np.zeros(np.shape(x_true));
x_smooth = np.zeros(np.shape(x_true));
P_negative = np.zeros((tmax,nx,nx));
P_positive = np.zeros((tmax,nx,nx));

x_true[:,0] = np.array([1,1,0,1.5]);

P_positive[0] = np.eye(nx)*10;
P_negative[0] = np.eye(nx)*10;
R = 0.03; sigmav = R**2;
Q = np.eye(2)*20;sigmaw = Q**2

t = np.linspace(0,tmax*T,tmax)

for i in range(1,tmax):
    x_true[:,i] = F.dot(x_true[:,i-1]) +Gamma.dot(np.random.normal(0,np.sqrt(Q)).dot(np.ones([2,1]))).T    
    z[:,i] = H.dot(x_true[:,i]) + np.random.normal(0,sigmav,2)
    
    #time update
    x_priori[:,i] = np.dot(F,x_estimate[:,i-1]);
    P_negative[i] = F.dot(P_positive[i-1]).dot(F.T) + Gamma.dot(Q).dot(Gamma.T)
        
    K = P_negative[i].dot(H.T).dot(np.linalg.inv(H.dot(P_negative[i]).dot(H.T) + R))
    #K = P_positive.dot(H.T)/((H.dot(P_positive).dot(H.T) + R))
    P_positive[i] = (np.eye(np.size(P_negative[i],0)) - K.dot(H))*(P_negative[i])
    x_estimate[:,i] = x_priori[:,i] + K.dot(z[:,i] - H.dot(x_priori[:,i]))
    x_smooth[:,i] = x_estimate[:,i]
    K_smooth =  P_positive[i-1].dot(F.T).dot(np.linalg.inv(P_negative[i]));
    x_smooth [:,i] = x_estimate[:,i] + K_smooth.dot(x_smooth[:,i] -x_priori[:,i])

#x_smooth[:,49] = x_estimate[:,49]
[P_backpass,x_backpass] = backPass(F,P_negative,P_positive,x_priori,x_estimate)
    
    #measurement update
    
    #print(x_priori)
    
plot.figure(1)
plot.subplot(1,2,1)
plot.title("Position")
plot.plot(z[0,:],z[1,:],'r.'
          ,x_estimate[0,:],x_estimate[2,:],'b')
plot.subplot(1,2,2)
plot.title("Velocity")
plot.plot(t,x_true[3,:],'r',t,x_estimate[3,:],'b')
plot.legend(["measurements",'position estimate'])

plot.figure(2)
plot.subplot(1,2,1)
plot.title("Position")
plot.plot(z[0,:],z[1,:],'r.'
          ,x_backpass[0,:],x_backpass[2,:],'b')
plot.legend(["measurments",'smoothed'])
plot.subplot(1,2,2)
plot.title("Velocity")
plot.plot(t,x_true[3,:],'r',t,x_backpass[3,:],'b')
plot.legend(["true",'smoothed'])


plot.figure(3)
plot.subplot(1,2,1)
plot.title("Position")
plot.plot(z[0,:],z[1,:],'r.'
          ,x_smooth[0,:],x_smooth[2,:],'b')
plot.legend(["measurments",'smoothed'])
plot.subplot(1,2,2)
plot.title("Velocity")
plot.plot(t,x_true[3,:],'r',t,x_smooth[3,:],'b')
plot.legend(["true",'smoothed'])
