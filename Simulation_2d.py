#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 09:20:46 2019

@author: csteed
"""
import os as os;
import numpy as np;
import matplotlib.pyplot as plot;
import filterpy.kalman as kfilter;

T = 1;
tmax = 50;
F = np.array([[1, T],[0, 1]]);np.zeros(np.shape(F));
F = np.block([
        [F, np.zeros(np.shape(F))],
        [np.zeros(np.shape(F)),F]
        ]);

V2 = np.block([[T**2/2, 0],
               [T, 0]
               ])
V2 = np.vstack([
        V2, 
        V2[:,[1,0]], 
        
        ])
F_cv = np.block([
        [1, T, T**2/2],
        [0, 1, T],
        [0, 0, 1]
        ]);
F_cv= np.block([
        [F, V2],
        [np.zeros([V2.shape[1], F.shape[0]]), np.eye(2)]
        ]);

Gamma = np.array([[pow(T,2)/2],[T]]);
Gamma = np.block([
        [Gamma, np.zeros(np.shape(Gamma))],
        [np.zeros(np.shape(Gamma)),Gamma]
        ]);
Gamma_ca = np.array([
        [T**3/6], 
        [T**2/2]
        ]);
Gamma_ca = np.block([
        [Gamma_ca, np.zeros(np.shape(Gamma_ca))],
        [np.zeros(np.shape(Gamma_ca)), Gamma_ca]
        ]);
Gamma_ca = np.vstack([
        Gamma_ca,
        np.eye(2)
        ])

H = np.array([[1,0,0,0],[0,0,1,0]]);
H2 = np.array([[1,0,0,0,0,0],[0,0,1,0,0,0]]);
z = np.zeros([2,tmax])

x_true = np.zeros([4,tmax]);
nx = np.size(x_true,0);
x_estimate = np.zeros(np.shape(x_true));
x_priori = np.zeros(np.shape(x_true));
x_smooth = np.zeros(np.shape(x_true));
P_negative = np.zeros((tmax,nx,nx));
P_positive = np.zeros((tmax,nx,nx));

x_true[:,0] = np.array([1,1,0,1.5]);
x_estimate[:,0] = np.array(x_true[:,0]-1)
P_positive[0] = np.eye(nx)*1;
P_negative[0] = np.eye(nx)*1;
sigmav =np.diag([1,1])*30;
sigmaW =np.array([1,1])*2;
R = sigmav**2;
Q = np.diag(sigmaW**2);
Q2 = np.diag([1,1,1]);

t = np.linspace(0,tmax*T,tmax)

constant_velocity = kfilter.KalmanFilter(dim_x=4, dim_z=2);
constant_velocity.F = F;
constant_velocity.H = H;
constant_velocity.Q = Gamma.dot(Q).dot(Gamma.T);
constant_velocity.R = R;
constant_velocity.x =  x_true[:,0]
constant_velocity.P = P_positive[0];



constant_acceleration = kfilter.KalmanFilter(dim_x=6, dim_z=2);
constant_acceleration.F = F_cv;
constant_acceleration.H = H2;
constant_acceleration.Q = Gamma_ca.dot(Q).dot(Gamma_ca.T);
constant_acceleration.R = R;

constant_acceleration.x = np.array([1, 0, 0, 0, 0, 0])*1.0;
constant_acceleration.P = np.diag([2**2, 2**2]*2 + [2**2]*2);


filters = [constant_velocity, constant_acceleration]
mu_input = [0.5, 0.5]  # each filter is equally likely at the start
trans = np.array([[0.98, 0.02], [0.02, 0.98]])
imm = kfilter.IMMEstimator(filters, mu_input, M=trans);

x_imm = np.zeros([tmax,6])
P_imm = np.zeros([tmax,6,6])
P_imm[0] = np.diag([100**2, 20**2]*2 + [20**2]*2);
mu_store = np.zeros([tmax,len(imm.filters)])
mu_store[0,:] = mu_input

for i in range(1,tmax):
    #generate true dynamics and measurement
    x_true[:,i] = F.dot(x_true[:,i-1]) +Gamma.dot(np.random.normal(0,sigmaW))
    z[:,i] = H.dot(x_true[:,i]) + np.random.normal(0,1,2).dot(sigmav)    
#    imm.predict();
#    imm.update(z[:,i]);
#    x_imm[i,:] = imm.x;
#    P_imm[i] = imm.P
#    mu_store[i] = imm.mu;

x_imm_filtered,_,mu_store,_,_,_,_,_,_,_ = imm.batch_filter(z.T)
constant_velocity.x =  x_true[:,0].T
constant_velocity.P = P_positive[0];
constant_acceleration.x = np.array([0, 0, 0, 0, 0, 0])*1.0;
constant_acceleration.P = np.diag([2**2, 2**2]*2 + [2**2]*2);
x_imm_smoothed = imm.batch_smooth(z.T)[0]

constant_velocity.x =  x_true[:,0].T
constant_velocity.P = P_positive[0];
[x_cv_filtered,P_positive,_,_] = constant_velocity.batch_filter(z.T)    
[x_cv_smoothed,_,_,_] = constant_velocity.rts_smoother(x_cv_filtered, P_positive)
#
constant_acceleration.x = np.array([0, 0, 0, 0, 0, 0])*1.0;
constant_acceleration.P = np.diag([2**2, 2**2]*2 + [2**2]*2);
[x_ca_filtered,P_a,_,_] = constant_acceleration.batch_filter(z.T);
[x_ca_smoothed,_,_,_] = constant_acceleration.rts_smoother(x_ca_filtered, P_a);

#[imm_smooth_cv,_,_,_] = constant_velocity.rts_smoother(x_imm[:,0:4],P_imm[:,:-2,:-2] )
#[imm_smooth_ca,_,_,_] = constant_acceleration.rts_smoother(x_imm, P_imm )


save_location ='./figs/'+os.path.basename(__file__)

plot.figure()

plot.gcf().suptitle("Position filtered")
plot.plot(z[0,:],z[1,:],'r.',
          x_cv_filtered[:,0], x_cv_filtered[:,2], 'b',
          x_ca_filtered[:,0], x_ca_filtered[:,2], 'k',
          x_imm_filtered[:,0], x_imm_filtered[:,2], 
          )
plot.legend(["measurments",'cv','ca', 'imm'])
plot.savefig(save_location+"_"+plot.gcf()._suptitle.get_text()+'.png', dpi=300)

plot.figure()
plot.gcf().suptitle("Velocity filtered")
plot.plot(t,x_true[3,:],'r',
          t,x_cv_filtered[:,3].T,'b',
          t,x_ca_filtered[:,3],'k',
          t,x_imm_filtered[:,3],
          )
plot.legend(["true",'cv','ca','imm'])
plot.savefig(save_location+"_"+plot.gcf()._suptitle.get_text()+'.png', dpi=300)

plot.figure()
plot.gcf().suptitle("Position smoothed")
plot.plot(x_true[0,:],x_true[2,:],'r.',
          x_cv_smoothed[:,0], x_cv_smoothed[:,2], 'b',
          x_ca_smoothed[:,0], x_ca_smoothed[:,2], 'k',
          x_imm_smoothed[:,0], x_imm_smoothed[:,2]
          )
plot.legend(["true",'cv','ca','imm'])
plot.savefig(save_location+"_"+plot.gcf()._suptitle.get_text()+'.png', dpi=300)

plot.figure()
plot.gcf().suptitle("Velocity smoothed")
plot.plot(t, x_true[3,:],'r',
          t, x_cv_smoothed[:,3].T,'b',
          t, x_ca_smoothed[:,3],'k',
          t, x_imm_smoothed[:,3]
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




