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

T = 2;
t_final = 100*T;
n_dimentions = 2;             
F = np.block([               
    [1, T],
    [0, 1]
    ]);
F2 = np.block([
        [1, T, T^2],
        [0, 1, T],
        [0, 0, 1]
        ]);

Gamma = np.array([[T**2/2], [T]]);      # Gamma (related to system noise)
Gamma2  = np.array([[T**3/6], [T**2/2], [T]]); 
#Gamma = np.array([1,0]);
H = np.atleast_2d([1,0]);          # Measurement matrix: ONLY POSITION
## Initial Conditions
x_true = np.zeros([n_dimentions,t_final]);
xp = np.zeros(np.shape(x_true))
y = np.zeros([t_final-1])

x_true[:,0] = np.array([0,5]);     # true initial state
xp[:,0] = np.array([1,3]);    # guess of initial posteriori estimation
Pp = np.block([              # guess of initial error covariance
    [100**2, 0],
    [0, 20**2]
    ]);
sigma_Pp = np.zeros([t_final,n_dimentions,n_dimentions])
sigma_Pp[1] = np.sqrt(np.diag(Pp));
## Noise
sigma_w = np.array([2]);        # system noise (std of acceleration) 
sigma_v = 30;       # measurement noise (std of position sensor)
Q = np.diag(np.power(sigma_w,2));      # system noise covariance matrix
R = np.power(sigma_v,2);      # measurement noise covariance matrix

Constant_velocity  = pk.KalmanFilter(transition_matrices= F,
                                     observation_matrices= H,
                                     transition_covariance= Gamma.dot(Q).dot(Gamma.T),
                                     observation_covariance= R,
                                     initial_state_covariance= Pp,
                                     initial_state_mean= xp[:,0]
                                     )
Constant_acceleration_filter = pk.KalmanFilter(transition_matrices=F2,
                                               observation_matrices= np.array([1, 0, 0]),
                                               transition_covariance= Gamma2.dot(Q).dot(Gamma2.T),
                                               observation_covariance=R, 
                                               initial_state_covariance= np.diag([100**2, 20**2, 2**2]),
                                               initial_state_mean= [1, 0, 0])
for i in range(0,t_final-1):
    ## True dynamics
    x_true[:,i+1] = F.dot(x_true[:,i]) +Gamma.dot(np.random.normal(0,sigma_w))  # system dynamics    

    [x_, P_, K] =kf.timeUpdate(F,H,R,Q,Gamma,xp[:,i],Pp);   
    y[i] = x_true[0,i+1] + np.random.normal(0,sigma_v);
    [xp[:,i+1], Pp] = kf.measurementUpdate(P_,K,x_,H,y[i])
    Pp = np.dot((np.eye(np.size(K.dot(H),0))- K.dot(H)), (P_))
    xp[:,i+1] = x_ + K.dot(y[i] - H.dot(x_))
    
x_filtered = Constant_velocity.filter(y)[0]
x_smoothed = Constant_velocity.smooth(y)[0]

xa_filtered = Constant_acceleration_filter.filter(y)[0]
xa_smooth = Constant_acceleration_filter.smooth(y)[0]
#    
plot.figure(1)
plot.subplot(1,2,1)
plot.title("Position")
plot.plot(y,'r.'
          ,xp[0,:],'b')
plot.legend(["measurements",'position estimate'])

plot.subplot(1,2,2)
plot.title("Velocity")
plot.plot(x_true[1,:],'r',xp[1,:],'b')
plot.legend(["v_true",'v_filter'])

plot.figure(2)
plot.subplot(1,2,1)
plot.title("Position")
plot.plot(y,'r.'
          ,x_filtered[:,0],'b')
plot.legend(["measurements",'position estimate'])

plot.subplot(1,2,2)
plot.title("Velocity")
plot.plot(x_true[1,:],'r',x_filtered[:,1],'b')
plot.legend(["v_true",'v_filter'])

plot.figure(3)
plot.subplot(1,3,1)
plot.title("Position CA")
plot.plot(y,'r.',
          xa_filtered[:,0],'b',
          xa_smooth[:,0],'k',)
plot.legend(["measurements",'position estimate'])

plot.subplot(1,3,2)
plot.title("Velocity CA")
plot.plot(x_true[1,:],'r',
          xa_filtered[:,1],'b',
          xa_smooth[:,1],'k',
          )
plot.legend(["v_true",'v_filter'])

plot.subplot(1,3,3)
plot.title("Acceleration")
plot.plot(xa_filtered[:,2],'b',
          xa_smooth[:,2],'k',)
plot.legend(["v_true",'v_filter'])

plot.figure(4)
plot.title("Velocity CV vs CA")
plot.plot(x_true[1,:],'r',
          x_smoothed[:,1],'b',
          xa_smooth[:,1],'k',
          )
plot.legend(["v_true",'v_filter','v_ca'])