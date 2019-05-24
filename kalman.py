#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 13:12:39 2019

@author: csteed
"""
import numpy as np;
import pykalman as pkal;

def measurementUpdate(Pp,K,x_prior,H,z):
        Pp = (np.eye(np.size(Pp,0)) - K.dot(H))*(Pp)
        x_estimate = x_prior + K.dot(z - H.dot(x_prior))
        return (Pp, x_estimate)

def backPass(F, Gamma, Q, P_filt, x_filt):
    x_smooth = np.zeros(np.shape(x_filt));
    P_smooth = np.zeros(np.shape(P_filt));
    
    x_smooth[:,-1] = x_filt[:,-1];
    P_smooth[-1] = P_filt[-1];

    t = np.size(x_filt,1);
    for i in range(t-1,0,-1):
        P_pred = F.dot(P_filt[i-1]).dot(F.T) + Gamma.dot(Q).dot(Gamma.T)
        
        L = P_filt[i-1].dot(F.T).dot(np.linalg.inv(P_pred));
        x_pred = F.dot(x_filt[:,i-1])
        x_smooth[:,i-1] = x_filt[:,i-1] + L.dot(x_smooth[:,i] - x_pred);
        
        P_smooth[i-1] = P_filt[i-1] + L.dot(P_smooth[i]-P_pred).dot(L.T);
        
        [x_smooth[:,i-1],P_smooth[i-1],v] = pkal.standard._smooth_update(F, x_filt[:,i-1], P_filt[i-1], x_pred, P_pred, x_smooth[:,1], P_smooth[i])
    return (P_smooth,x_smooth);

