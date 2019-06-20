#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 13:12:39 2019

@author: csteed
"""
import numpy as np;

def measurementUpdate(P_pred,K,x_prior,H,z):

    Pp = (np.eye(np.size(K.dot(H),0)) - K.dot(H)).dot(P_pred)
    x_estimate = x_prior + K.dot(z - H.dot(x_prior))
    return (x_estimate, Pp)
    

def timeUpdate(F,H,R,Q,Gamma,x_filter_prev,P_filter_prev):
    
    x_predicted = np.dot(F,x_filter_prev);
    P_predicted = F.dot(P_filter_prev).dot(F.T) + Gamma.dot(Q).dot(Gamma.T)
    if(np.size(R) == 1):
        Kalman_gain = P_predicted.dot(H.T).dot(1/(H.dot(P_predicted).dot(H.T) + R))
    else:
        Kalman_gain = P_predicted.dot(H.T).dot(np.linalg.pinv((H.dot(P_predicted).dot(H.T) + R)))
    return(x_predicted, P_predicted, Kalman_gain)

def backPass(F, Gamma, Q, P_filt, x_filt):
    x_smooth = np.zeros(np.shape(x_filt));
    P_smooth = np.zeros(np.shape(P_filt));
    
    x_smooth[:,-1] = x_filt[:,-1];
    P_smooth[-1] = P_filt[-1];
    t = np.size(x_filt,1);
    
    for i in range(t-1,0,-1):
        P_pred = F.dot(P_filt[i-1]).dot(F.T) + Gamma.dot(Q).dot(Gamma.T);
        x_pred = F.dot(x_filt[:,i-1]);
        L = P_filt[i-1].dot(F.T).dot(np.linalg.pinv(P_pred));
        x_smooth[:,i-1] = x_filt[:,i-1] + L.dot(x_smooth[:,i] - x_pred);
        P_smooth[i-1] = P_filt[i-1] + L.dot(P_smooth[i]-P_pred).dot(L.T);
    return (x_smooth, P_smooth);

    
