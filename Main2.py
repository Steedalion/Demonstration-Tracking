#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 16:33:55 2019

@author: csteed
"""

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
    keyPressed = cv2.waitKey(200)
    if (keyPressed == ord('q')):
        break;
    if (keyPressed == ord('p')):
        cv2.waitKey(0);
cv2.destroyAllWindows()



print("Filter entered")

import numpy as np;
import matplotlib.pyplot as plot;
import kalman_lib as kf; 
import pykalman as pkal;

    

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
z = np.array(points).T
tmax = len(z[1]);
x_true = np.zeros([4,tmax]);
nx = np.size(x_true,0);
x_estimate = np.zeros(np.shape(x_true));   
x_estimate[:,0] = np.array([z[0][0],1,z[1][0],0])   
P_filtered = np.zeros((tmax,nx,nx));
P_filtered[0] = np.eye(np.size(x_true,0));
R = 0.002;
sigmav =np.diag([1,1])*0.01;
sigmaW =np.diag([1,1])*10;
R = sigmav**2;
Q = sigmaW**2;
t = np.linspace(0,tmax*T,tmax)

constant_velocity = pkal.KalmanFilter(transition_matrices=F,
                                      observation_matrices=H,
                                      transition_covariance=Gamma.dot(Q).dot(Gamma.T),
                                      observation_covariance=R,
                                      initial_state_mean=x_estimate[:,0],
                                      initial_state_covariance=P_filtered[0])
F2 = np.block([
        [1, T, T**2],
        [0, 1, T],
        [0, 0, 1]
        ]);
F2 = np.block([
        [F2, np.zeros(np.shape(F2))],
        [np.zeros(np.shape(F2)),F2]
        ]);
Gamma2  = np.array([[T**3/6], [T**2/2], [T]]);
Gamma2 = np.block([
        [Gamma2, np.zeros(np.shape(Gamma2))],
        [np.zeros(np.shape(Gamma2)), Gamma2]
        ]);    
H2 = np.array([[1,0,0,0,0,0],[0,0,0,1,0,0]]);
sigmaW2 =np.diag([1,1,1])*10;
Q2 = sigmaW2**2;
constant_acceleration = pkal.KalmanFilter(transition_matrices=F2,
                                          observation_matrices=H2,
                                          transition_covariance=Gamma2.dot(Q).dot(Gamma2.T),
                                          observation_covariance=R,
                                          initial_state_mean=np.zeros([6]),
                                          initial_state_covariance=np.diag([1,1,1,1,1,1])
                                          )

for i in range(1,tmax):   
    #time update
    [x_priori, P_pred, K] = kf.timeUpdate(F,H,R,Q,Gamma,x_estimate[:,i-1],P_filtered[i-1])
    #K = P_filtered.dot(H.T)/((H.dot(P_filtered).dot(H.T) + R))
    #measurement update
    [x_estimate[:,i], P_filtered[i]] = kf.measurementUpdate(P_pred,K,x_priori,H,z[:,i])
    

x_cv = constant_velocity.filter(z.T)[0]    
x_ca = constant_acceleration.filter(z.T)[0]    
[x_backpass, P_backpass] = kf.backPass(F,Gamma,Q,P_filtered,x_filt=x_estimate)

plot.figure(1)
plot.subplot(1,2,1)
plot.title("Position")
plot.plot(z[0,:],z[1,:],'r.'
          ,x_estimate[0,:],x_estimate[2,:],'b')
#plot.axis([0,300,0,200])
plot.legend(["measurements",'position estimate'])

plot.subplot(1,2,2)
plot.title("Velocity")
plot.plot(t,x_estimate[1,:],'b',
          t,x_estimate[3,:],'r')
plot.legend(["v_x",'v_y'])

plot.figure(2)
plot.subplot(1,2,1)
plot.title("Position")
plot.plot(z[0,:],z[1,:],'r.'
          ,x_backpass[0,:],x_backpass[2,:],'b')
plot.legend(["measurements",'position estimate'])

plot.subplot(1,2,2)
plot.title("Velocity")
plot.plot(t,x_backpass[1,:],'b',
          t,x_backpass[3,:],'r')
plot.legend(["v_x",'v_y'])

plot.figure(3)
plot.subplot(1,2,1)
plot.title("Position pykal")
plot.plot(z[0,:],z[1,:],'r.'
          ,x_cv[:,0],x_cv[:,2],'b',
          x_ca[:,0], x_ca[:,3])
plot.legend(["measurements",'position estimate_cv','position ca'])

plot.subplot(1,2,2)
plot.title("Velocity pykal")
plot.plot(t,x_cv[:,1],'b',
          t,x_cv[:,3],'r',
          t,x_ca[:,1],
          t,x_ca[:,4])
plot.legend(["v_x",'v_y'])
plot.plot(t,x_cv[:,1],'b',
          t,x_cv[:,3],'r')
plot.legend(["v_x",'v_y','vx_ca','vy_ca'])
    
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
