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

