#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 16:33:55 2019

@author: csteed
"""
import os as os;

import cv2
import numpy as np
from collections import deque


greenLower = (6, 120, 100)
greenUpper =  (64, 255, 255)

greenLower = (8, 12, 20)
greenUpper =  (70, 255, 100)

fps=30;
period=1/fps;

video = cv2.VideoCapture('Blue_ball_straight.mp4')
#video = cv2.VideoCapture('ball_video.mp4')
frameCount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
frameWidth = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

fc = 0
ret = True

while (fc < frameCount  and ret):
    ret, buf[fc] = video.read()
    fc += 1

points = deque(maxlen=len(buf))

video.release() 
i=0; 
while (i<len(buf)):
    
    currentFrame = buf[i];
    i=i+1;
    
    
    blurred = cv2.GaussianBlur(currentFrame, (11, 11), 0);
    hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV);
    mask = cv2.inRange(hsv, greenLower, greenUpper);
    mask = cv2.erode(mask, None, iterations=2);
    mask = cv2.dilate(mask, None, iterations=2);
    ballContours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);

    ballContours = ballContours[-2]; #extracting contours;

    
    if len(ballContours) > 0:
        largestContour = max(ballContours, key=cv2.contourArea);
        ((circleX, circleY),circleRadius) = cv2.minEnclosingCircle(largestContour);
        cv2.circle(currentFrame, (int(circleX),int(circleY)), int(circleRadius),(0,255,255),2)
        points.appendleft((int(circleX),int(circleY)))
        #print("points: ",points);
    #Draw trail line    
    for j in range(1,len(points)):
        cv2.line(currentFrame,points[j - 1], points[j], (0,0,255), 3)
    if (0 == i%10):
        cv2.imwrite('./gif/'+'pic'+str(i)+'.png',currentFrame)
    cv2.namedWindow('frame 10');
    cv2.imshow('frame 10', currentFrame);
    cv2.namedWindow('edited')
    cv2.imshow('edited', hsv)
    cv2.namedWindow('mask')
    cv2.imshow('mask', mask)
    keyPressed = cv2.waitKey(10)
    if (keyPressed == ord('q')):
        break;
    if (keyPressed == ord('p')):
        cv2.waitKey(0);
cv2.destroyAllWindows()



print("Filter entered")

import numpy as np;
import matplotlib.pyplot as plot;
import filterpy.kalman as kfilter;

    

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
z = np.array(points).T
tmax = len(z[1]);
x_true = np.zeros([4,tmax]);
nx = np.size(x_true,0);


R = 0.002;
sigmav =np.diag([1,1])*0.01;
sigmaW =np.diag([1,1])*10;
R = sigmav**2;
Q = sigmaW**2;
t = np.linspace(0,tmax*T,tmax)


constant_velocity = kfilter.KalmanFilter(dim_x=4, dim_z=2);
constant_velocity.F = F;
constant_velocity.H = H;
constant_velocity.Q = Gamma.dot(Q).dot(Gamma.T);
constant_velocity.R = R;
constant_velocity.x =  np.zeros(x_true[:,0].shape);
constant_velocity.P = np.eye(np.size(x_true,0));

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
    


H2 = np.array([[1,0,0,0,0,0],[0,0,1,0,0,0]]);
sigmaW2 =np.diag([1,1,1])*10;
Q2 = sigmaW2**2;


constant_acceleration = kfilter.KalmanFilter(dim_x=6, dim_z=2);
constant_acceleration.F = F2;
constant_acceleration.H = H2;
constant_acceleration.Q = Gamma_ca.dot(Q).dot(Gamma_ca.T);
constant_acceleration.R = R;

#constant_acceleration.x = np.array([1, 0, 0, 0, 0, 0])*1.0;
#constant_acceleration.P = np.diag([100**2, 20**2]*2 + [20**2]*2);
constant_acceleration.x = np.array([0, 0, 0, 0, 0, 0])*1.0;
constant_acceleration.P = np.eye(6);

x_ca_filtered, P_ca = constant_acceleration.batch_filter(z.T)[0:2]
x_ca_smoothed = constant_acceleration.rts_smoother(x_ca_filtered,P_ca)[0];
    
save_location ='./figs/'+os.path.basename(__file__)

#imm
filters = [constant_velocity, constant_acceleration]
mu_input = [0.5, 0.5]  # each filter is equally likely at the start
trans = np.array([[0.98, 0.02], [0.02, 0.98]])
imm = kfilter.IMMEstimator(filters, mu_input, M=trans);

x_imm_filtered = imm.batch_filter(z.T)[0];
x_imm_smoothed = imm.batch_smooth(z.T)[0];

constant_velocity.x =  np.zeros(x_true[:,0].shape);
constant_velocity.P = np.eye(np.size(x_true,0));
x_cv_filtered,P_cv = constant_velocity.batch_filter(z.T)[0:2]    
x_cv_smoothed = constant_velocity.rts_smoother(x_cv_filtered, P_cv)[0];   



plot.figure()
plot.gcf().suptitle("Position estimation")
plot.plot(z[0,:],z[1,:],'r.',
          x_ca_filtered[:,0], x_ca_filtered[:,2], 'k',
          x_ca_smoothed[:,0], x_ca_smoothed[:,2], 'b', 
          )

plot.figure()
plot.gcf().suptitle("Velocity estimation")
plot.plot(t,x_ca_filtered[:,1],
          t,x_ca_smoothed[:,1],'k',
          t,x_ca_filtered[:,3],'k',
          t,x_ca_smoothed[:,3],'k',
          )
plot.legend(["x filtered","x smoothed","y filtered","y smoothed"])
plot.savefig(save_location+"_"+plot.gcf()._suptitle.get_text()+'.png', dpi=300)

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
plot.gcf().suptitle("Velocity x filtered")
plot.plot(
          t,x_cv_filtered[:,1].T,'b',
          t,x_ca_filtered[:,1],'k',
          t,x_imm_filtered[:,1],
          )
plot.legend(['cv','ca','imm'])
plot.savefig(save_location+"_"+plot.gcf()._suptitle.get_text()+'.png', dpi=300)

plot.figure()
plot.gcf().suptitle("Velocity y filtered")
plot.plot(
          t,x_cv_filtered[:,3].T,'b',
          t,x_ca_filtered[:,3],'k',
          t,x_imm_filtered[:,3],
          )
plot.legend(['cv','ca','imm'])
plot.savefig(save_location+"_"+plot.gcf()._suptitle.get_text()+'.png', dpi=300)


plot.figure()
plot.gcf().suptitle("Position smoothed")
starting = 3
plot.plot(
          x_cv_smoothed[starting:,0], x_cv_smoothed[starting:,2], 'b',
          x_ca_smoothed[starting:,0], x_ca_smoothed[starting:,2], 'k',
          x_imm_smoothed[starting:,0], x_imm_smoothed[starting:,2]
          )
plot.legend(['cv','ca','imm'])
plot.savefig(save_location+"_"+plot.gcf()._suptitle.get_text()+'.png', dpi=300)

starting = 5
plot.figure()
plot.gcf().suptitle("Velocity smoothed")
plot.plot(
          t[starting:], x_cv_smoothed[starting:,3].T,'b',
          t[starting:], x_ca_smoothed[starting:,3],'k',
          t[starting:], x_imm_smoothed[starting:,3]
          )
plot.legend(['cv','ca', 'imm'])
plot.savefig(save_location+"_"+plot.gcf()._suptitle.get_text()+'.png', dpi=300)
    
points_kf = tuple(map(tuple,np.vstack([x_estimate[0].T,x_estimate[2]]).T.astype(int)))
points_bp = tuple(map(tuple,np.vstack([x_backpass[0].T,x_backpass[2]]).T.astype(int)))


for j in range(1,len(points)):
        cv2.line(currentFrame,points_kf[j - 1], points_kf[j], (0,255,0), 2)
        cv2.line(currentFrame,points_bp[j - 1], points_kf[j], (255,0,0), 1)
cv2.namedWindow('frame 10');
cv2.imshow('frame 10', currentFrame);
cv2.imwrite('ball tracking.png',currentFrame)
keyPressed = cv2.waitKey(3000)
cv2.destroyAllWindows()
