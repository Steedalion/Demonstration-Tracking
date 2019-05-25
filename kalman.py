#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 13:12:39 2019

@author: csteed
"""
import numpy as np;
import pykalman as pkal;

def measurementUpdate(P_pred,K,x_prior,H,z):
        Pp = (np.eye(np.size(P_pred,0)) - K.dot(H))*(P_pred)
        x_estimate = x_prior + K.dot(z - H.dot(x_prior))
        return (Pp, x_estimate)

def timeUpdate(F,H,R,Q,Gamma,x_filter_prev,P_filter_prev):
    x_predicted = np.dot(F,x_filter_prev);
    P_predicted = F.dot(P_filter_prev).dot(F.T) + Gamma.dot(Q).dot(Gamma.T)
    Kalman_gain = P_predicted.dot(H.T).dot(np.linalg.inv(H.dot(P_predicted).dot(H.T) + R))
    return(x_predicted,P_predicted,Kalman_gain)

def backPass(F, Gamma, Q, P_filt, x_filt):
    x_smooth = np.zeros(np.shape(x_filt));
    P_smooth = np.zeros(np.shape(P_filt));
    
    x_smooth[:,-1] = x_filt[:,-1];
    P_smooth[-1] = P_filt[-1];

    t = np.size(x_filt,1);
    for i in range(t-1,0,-1):
        P_pred = F.dot(P_filt[i-1]).dot(F.T) #+ Gamma.dot(Q).dot(Gamma.T);
        x_pred = F.dot(x_filt[:,i-1]);
        
        L = P_filt[i-1].dot(F.T).dot(np.linalg.pinv(P_pred));
        
        L = (
        np.dot(P_filt[i-1],
               np.dot(F.T,
                      np.linalg.pinv(P_pred)))
    )
        
        x_smooth[:,i-1] = x_filt[:,i-1] + L.dot(x_smooth[:,i] - x_pred);
        
        x_smooth[:,i-1] = (
        x_filt[:,i-1]
        + np.dot(L,
                 x_smooth[:,i] - x_pred)
        )
        
        P_smooth[i-1] = P_filt[i-1] + L.dot(P_smooth[i]-P_pred).dot(L.T);
        
        P_smooth[i-1] = (
        P_filt[i-1]
        + np.dot(L,
                 np.dot(
                    (P_smooth[i]
                        - P_pred),
                    L.T
                 ))
    )
        
       # [x_smooth[:,i-1],P_smooth[i-1],v] = pkal.standard._smooth_update(F, x_filt[:,i-1], P_filt[i-1], x_pred, P_pred, x_smooth[:,i], P_smooth[i])
    return (P_smooth,x_smooth);

#def _smooth_update(transition_matrix, filtered_state_mean,
#                   filtered_state_covariance, predicted_state_mean,
#                   predicted_state_covariance, next_smoothed_state_mean,
#                   next_smoothed_state_covariance):
#    kalman_smoothing_gain = (
#        np.dot(filtered_state_covariance = P_filt[i-1],
#               np.dot(transition_matrix.T = F.T,
#                      linalg.pinv(predicted_state_covariance)))
#    )
#
#    smoothed_state_mean = (
#        filtered_state_mean
#        + np.dot(kalman_smoothing_gain,
#                 next_smoothed_state_mean - predicted_state_mean)
#    )
#    smoothed_state_covariance = (
#        filtered_state_covariance
#        + np.dot(kalman_smoothing_gain,
#                 np.dot(
#                    (next_smoothed_state_covariance
#                        - predicted_state_covariance),
#                    kalman_smoothing_gain.T
#                 ))
#    )