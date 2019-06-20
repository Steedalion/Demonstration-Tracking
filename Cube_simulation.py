#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 08:59:49 2019

@author: csteed
"""
import os as os;
import numpy as np;
import matplotlib.pyplot as plot;
import filterpy.kalman as kfilter;

T = 1;
t_period = 10;
t_final = 3*t_period;
t = np.linspace(0,t_final*T,t_final)

F_system_cv = np.block([[1, T],
                 [0, 1]
                 ]);
F_0 = np.zeros(F_system_cv.shape)
F_I = np.eye(2)
F_system_ca = np.block([
        [F_system_cv, F_0, np.array([[T**2/2, 0],[T, 0]]) ],
        [F_0, F_I, F_0 ],
        [np.zeros([2,4]), np.array([[1,0],[0,0]])]
        ])

F_system_cv = np.block([
        [F_I, F_0, F_0],
        [F_0, F_system_cv, F_0],
        [F_0, F_0, F_0],
        ])
dim_x = F_system_cv.shape[0]

F = np.array([[1, T],[0, 1]]);
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
F2 = np.block([
        [1, T, T**2/2],
        [0, 1, T],
        [0, 0, 1]
        ]);
F2= np.block([
        [F, V2],
        [np.zeros([V2.shape[1], F.shape[0]]), np.eye(2)]
        ]);
H = np.array([[1,0,0,0],[0,0,1,0]]);
H2 = np.array([[1,0,0,0,0,0],[0,0,1,0,0,0]]);
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

z = np.zeros([2,100])

sigmav =np.diag([1,1])*30;
sigmaW =np.array([1,1])*2;
R = sigmav**2;
Q = np.diag(sigmaW**2);



x_true = np.zeros([t_final, dim_x])
x_true[0,:] = np.ones(dim_x)
zs = np.zeros([t_final,2])
constant_velocity = kfilter.KalmanFilter(dim_x=4, dim_z=2);
constant_velocity.F = F;
constant_velocity.H = H;
constant_velocity.Q = Gamma.dot(Q).dot(Gamma.T);
constant_velocity.R = R;
constant_velocity.x = np.ones([constant_velocity.dim_x])
constant_velocity.P = np.eye(4)*100;

constant_acceleration = kfilter.KalmanFilter(dim_x=6, dim_z=2);
constant_acceleration.F = F2;
constant_acceleration.H = H2;
constant_acceleration.Q = Gamma_ca.dot(Q).dot(Gamma_ca.T);
constant_acceleration.R = R;

constant_acceleration.x = x_true[0,:]
constant_acceleration.P = np.diag([100**2, 20**2]*2 + [20**2]*2);

filters = [constant_velocity, constant_acceleration]
mu_input = [0.5, 0.5]  # each filter is equally likely at the start
trans = np.array([[0.7, 0.3], [0.3, 0.7]])
imm = kfilter.IMMEstimator(filters, mu_input, M=trans);

for i in range(1,t_final):
    if (i<t_period):
        F_sys = F_system_cv;
    elif (i<=2*t_period):
        if (i == t_period):
            x_true[i-1,4] = 0.5
        F_sys = F_system_ca;
    else:
        x_true[i-1,3] = -1
        F_sys = F_system_cv
        
    x_true[i] = F_sys.dot(x_true[i-1])
    zs[i] = H2.dot(x_true[i])
    
x_imm_filtered,_,mu_store,_,_,_,_,_,_,_ = imm.batch_filter(zs)
x_imm_smoothed = imm.batch_smooth(z.T)[0]

constant_velocity.x = np.ones([constant_velocity.dim_x])
constant_velocity.P = np.eye(4)*100;
[x_cv_filtered,P_positive,_,_] = constant_velocity.batch_filter(zs)    
[x_cv_smoothed,_,_,_] = constant_velocity.rts_smoother(x_cv_filtered, P_positive)
#
constant_acceleration.x = x_true[0,:]
constant_acceleration.P = np.diag([100**2, 20**2]*2 + [20**2]*2);
[x_ca_filtered,P_a,_,_] = constant_acceleration.batch_filter(zs);
[x_ca_smoothed,_,_,_] = constant_acceleration.rts_smoother(x_ca_filtered, P_a);



save_location ='./figs/'+os.path.basename(__file__)

plot.figure()

plot.gcf().suptitle("Position filtered")
plot.plot(zs[:,0],zs[:,1],'r.',
          x_cv_filtered[:,0], x_cv_filtered[:,2], 'b',
          x_ca_filtered[:,0], x_ca_filtered[:,2], 'k',
          x_imm_filtered[:,0], x_imm_filtered[:,2], 
          )
plot.legend(["measurments",'cv','ca', 'imm'])
plot.savefig(save_location+"_"+plot.gcf()._suptitle.get_text()+'.png', dpi=300)

plot.figure()
plot.gcf().suptitle("Velocity filtered")
plot.plot(t,x_true[:,3],'r',
          t,x_cv_filtered[:,3],'b',
          t,x_ca_filtered[:,3],'k',
          t,x_imm_filtered[:,3],
          )
plot.legend(["true",'cv','ca','imm'])
plot.savefig(save_location+"_"+plot.gcf()._suptitle.get_text()+'.png', dpi=300)

plot.figure()
plot.gcf().suptitle("Position smoothed")
plot.plot(zs[:,0],zs[:,1],'r.',
          x_cv_smoothed[:,0], x_cv_smoothed[:,2], 'b',
          x_ca_smoothed[:,0], x_ca_smoothed[:,2], 'k',
          x_imm_smoothed[:,0], x_imm_smoothed[:,2]
          )
plot.legend(["true",'cv','ca','imm'])
plot.savefig(save_location+"_"+plot.gcf()._suptitle.get_text()+'.png', dpi=300)

plot.figure()
plot.gcf().suptitle("Velocity smoothed")
plot.plot(t, x_true[:,3],'r',
          t, x_cv_smoothed[:,3].T,'b',
          t, x_ca_smoothed[:,3],'k',
          #t, x_imm_smoothed[:,3]
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

error_cv_x, error_cv_y =[ zs[:,0] - x_cv_filtered[:,0], zs[:,1] - x_cv_filtered[:,2] ]
error_ca_x, error_ca_y =[ zs[:,0] - x_ca_filtered[:,0], zs[:,1] - x_ca_filtered[:,2] ]
error_imm_x, error_imm_y =[ zs[:,0] - x_imm_filtered[:,0], zs[:,1] - x_imm_filtered[:,2] ]

plot.figure()
plot.gcf().suptitle("Position error")
plot.plot(
        error_cv_x, error_cv_y,
        error_ca_x, error_ca_y,
        error_imm_x, error_imm_y,
          )
plot.legend(['cv','ca', 'imm'])
plot.savefig(save_location+"_"+plot.gcf()._suptitle.get_text()+'.png', dpi=300)