#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 11:38:19 2019

@author: csteed
"""
import numpy as np
import matplotlib.pyplot as plot;
import filterpy.kalman as kfilter;
import os as os;

T = 1;
t_final = 50*T;
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
x_imm = np.zeros([t_final,3])


x_true[:,0] = np.array([0,5]);     # true initial state
xp[:,0] = np.array([1,3]);    # guess of initial posteriori estimation
Pp = np.block([              # guess of initial error covariance
    [100**2, 0],
    [0, 20**2]
    ]);
sigma_Pp = np.zeros([t_final,n_dimentions,n_dimentions]);
P_post = np.zeros([t_final,n_dimentions,n_dimentions]);
sigma_Pp[1] = np.sqrt(np.diag(Pp));
## Noise
sigma_w = np.array([0.8]);        # system noise (std of acceleration) 
sigma_v = 30;       # measurement noise (std of position sensor)
Q = np.diag(np.power(sigma_w,2));      # system noise covariance matrix
R = np.power(sigma_v,2);      # measurement noise covariance matrix

constant_velocity = kfilter.KalmanFilter(dim_x=2, dim_z=1);
constant_velocity.F = F;
constant_velocity.x = x_true[:,0]
constant_velocity.H = H;
constant_velocity.Q = Gamma.dot(Q).dot(Gamma.T);
constant_velocity.P = Pp;
constant_velocity.R = R;

constant_acceleration = kfilter.KalmanFilter(dim_x=3, dim_z=1);
constant_acceleration.F = F2;
constant_acceleration.H = np.atleast_2d([1, 0, 0]);
constant_acceleration.Q = Gamma2.dot(Q).dot(Gamma2.T);
constant_acceleration.R = R;
constant_acceleration.x = np.array([1, 0, 0])*1.0;
constant_acceleration.P = np.diag([100**2, 20**2, 20**2]);




filters = [constant_acceleration, constant_velocity]
mu = [0.5, 0.5]  # each filter is equally likely at the start
trans = np.array([[0.97, 0.03], [0.03, 0.97]])
imm = kfilter.IMMEstimator(filters, mu, trans)
mu_store = np.zeros([t_final-1,len(imm.filters)])
for i in range(0,t_final-1):
    ## True dynamics
    x_true[:,i+1] = F.dot(x_true[:,i]) +Gamma.dot(np.random.normal(0,sigma_w))  # system dynamics    
    y[i] = x_true[0,i+1] + np.random.normal(0,sigma_v); # system measurement
    
#    [x_, P_, K] =kf.timeUpdate(F,H,R,Q,Gamma,xp[:,i],Pp);   
    
#    [xp[:,i+1], Pp] = kf.measurementUpdate(P_,K,x_,H,y[i])
#    Pp = np.dot((np.eye(np.size(K.dot(H),0))- K.dot(H)), (P_))
#    xp[:,i+1] = x_ + K.dot(y[i] - H.dot(x_))

    #imm.predict()
    #imm.update(y[i]) 
#    if (i == 0):
#        print();
#    else:
#        imm.smooth()
        #x_imm[i+1,:] = imm.x_smooth;
              
    
    
    #mu_store[i] = imm.mu
constant_velocity.x =  x_true[:,0]
constant_acceleration.x = np.array([1, 0, 0])*1.0;
imm.mu = [1.0, 0.0]

x_imm_filtered,_,mu_store,_,_,_,_,_,_,_ = imm.batch_filter(y)
x_imm_smoothed = imm.batch_smooth(y)[0]
constant_velocity.x =  x_true[:,0]
constant_velocity.P = Pp;
[x_cv_filtered,P_t,_,_] = constant_velocity.batch_filter(y)    
[x_cv_smoothed,_,_,_] = constant_velocity.rts_smoother(x_cv_filtered, P_t)

constant_acceleration.x = np.array([1, 0, 0])*1.0;
constant_acceleration.P = np.diag([100**2, 20**2, 2**2]);
[x_ca_filtered,P_a,_,_] = constant_acceleration.batch_filter(y);
[x_ca_smoothed,_,_,_] = constant_acceleration.rts_smoother(x_ca_filtered, P_a);

#imm
#    

save_location ='./figs/'+os.path.basename(__file__)

fig = plot.figure()
plot.gcf().suptitle("Positions filtered")
plot.plot(y,'r.',
          x_cv_filtered[:,0], 'b',
          x_ca_filtered[:,0], 'k',
          x_imm_filtered[:,0], 'y',
          )
plot.legend(["measurments",'cv','ca', 'imm'])
plot.savefig(save_location+"_"+plot.gcf()._suptitle.get_text()+'.png', dpi=300)

plot.figure()
plot.gcf().suptitle("Velocity filtered")
plot.plot(x_true[1,:],'r',
          x_cv_filtered[:,1],'b',
          x_ca_filtered[:,1],'k',
          x_imm_filtered[:,1],
          )
plot.legend(["true",'cv','ca','imm'])
plot.savefig(save_location+"_"+plot.gcf()._suptitle.get_text()+'.png', dpi=300)

plot.figure()
start = 2
plot.gcf().suptitle("Position smoothed")
plot.plot(x_true[0,(start+1):], 'r.',
          x_cv_smoothed[start:,0], 'b',
          x_ca_smoothed[start:,0],  'k',
          x_imm_smoothed[start:,0],
          )
plot.legend(["true",'cv','ca','imm'])
plot.savefig(save_location+"_"+plot.gcf()._suptitle.get_text()+'.png', dpi=300)

plot.figure()
plot.gcf().suptitle("Velocity smoothed")
start = 10
plot.plot(x_true[1,(start+1):],'r',
          x_cv_smoothed[start:,1],'b',
          x_ca_smoothed[start:,1],'k',
          x_imm_smoothed[start:,1]
          )
plot.legend(["true",'cv','ca', 'imm'])
plot.savefig(save_location+"_"+plot.gcf()._suptitle.get_text()+'.png', dpi=300)



plot.figure()
plot.gcf().suptitle("Mu")
plot.plot(
        mu_store
          
          )
plot.legend(["mu_ca",'mu_cv'])
plot.savefig(save_location+"_"+plot.gcf()._suptitle.get_text()+'.png', dpi=300)