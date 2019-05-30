#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 11:38:19 2019

@author: csteed
"""
import numpy as np
import kalman as kf;
import matplotlib.pyplot as plot;
import pykalman as pk;

T = 1;
t_final = 100;
n_dimentions = 2;             
F = np.block([               
    [1, T],
    [0, 1]
    ]);
Gamma = np.array([T**2/2, T]);      # Gamma (related to system noise)
Gamma = np.array([1,1]);
H = np.atleast_2d([1,0]);          # Measurement matrix: ONLY POSITION
## Initial Conditions
x = np.zeros([n_dimentions,t_final]);
y = np.zeros([t_final-1])
xp = np.zeros(np.shape(x))
x[:,0] = np.array([0,5]);     # true initial state
#x_i[:,1] = x[:,1];
xp[:,0] = np.array([1,3]);    # guess of initial posteriori estimation
Pp = np.block([              # guess of initial error covariance
    [100**2, 0],
    [0, 20**2]
    ]);
sigma_Pp = np.zeros([t_final,n_dimentions,n_dimentions])
sigma_Pp[1] = np.sqrt(np.diag(Pp));
## Noise
sigma_w = np.diag([1,1]);        # system noise (std of acceleration) 
sigma_v = 30;       # measurement noise (std of position sensor)
Q = np.power(sigma_w,2);      # system noise covariance matrix
R = np.power(sigma_v,2);      # measurement noise covariance matrix
#Y_p = np.eye(2)*1e-9;
#y_p(:,1) = Y_p*xp(:,1);
## KF Routine
Constant_velocity  = pk.KalmanFilter(transition_matrices= F,
                                     observation_matrices= H,
                                     transition_covariance= Q,
                                     observation_covariance= R,
                                     initial_state_covariance= Pp,
                                     initial_state_mean= xp[:,0].T
                                     )
for i in range(0,t_final-1,T):
    ## True dynamics
    x[:,i+1] = F.dot(x[:,i]) +Gamma.dot(np.random.normal(0,sigma_w))  # system dynamics    

    [x_, P_, K] =kf.timeUpdate(F,H,R,Q,Gamma,xp[:,i],Pp);   

    y[i] = x[0,i+1] + np.random.normal(0,sigma_v);
    [xp[:,i+1], Pp] = kf.measurementUpdate(P_,K,x_,H,y[i])
    Pp = np.dot((np.eye(np.size(K.dot(H),0))- K.dot(H)), (P_))
    xp[:,i+1] = x_ + K.dot(y[i] - H.dot(x_))
    
x_filtered = Constant_velocity.filter(y.T)[0]
#    
plot.figure(1)
plot.subplot(1,2,1)
plot.title("Position")
plot.plot(y,'r.'
          ,xp[0,:],'b')
plot.legend(["measurements",'position estimate'])

plot.subplot(1,2,2)
plot.title("Velocity")
plot.plot(x[1,:],'r',xp[1,:],'b')
plot.legend(["v_true",'v_filter'])

plot.figure(2)
plot.subplot(1,2,1)
plot.title("Position")
plot.plot(y,'r.'
          ,x_filtered[:,0],'b')
plot.legend(["measurements",'position estimate'])

plot.subplot(1,2,2)
plot.title("Velocity")
plot.plot(x[1,:],'r',x_filtered[:,1],'b')
plot.legend(["v_true",'v_filter'])


