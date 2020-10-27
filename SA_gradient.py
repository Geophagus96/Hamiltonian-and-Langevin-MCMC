# -*- coding: utf-8 -*-
"""
Created on Thu Oct 08 04:58:07 2020

@author: Yuze Zhou
"""

import numpy as np

def SA_two_way_interaction(a, W, n_samples, burn_in_samples):
    d = dict();
    n_pixels = W.shape[1]
    grad_a = np.matrix(np.zeros(n_pixels)) 
    grad_W = np.matrix(np.zeros([n_pixels,n_pixels]))
    s = np.matrix(np.zeros(n_pixels))
    for i in range (n_samples):
        s_propose=np.matrix(1*(np.random.rand(n_pixels)>0.5))
        potential_s = a*s.T+ 0.5*s*W*s.T
        potential_s_propose = a*s_propose.T+ 0.5*s_propose*W*s_propose.T
        if (potential_s_propose[0,0]>potential_s[0,0]):
            s = s_propose
        else:
            u = np.random.rand(1)
            p = np.exp(potential_s_propose-potential_s)
            if(p[0,0]<u[0]):
                s = s_propose
    for i in range (burn_in_samples):
        s_propose=np.matrix(1*(np.random.rand(n_pixels)>0.5))
        potential_s = a*s.T+ 0.5*s*W*s.T
        potential_s_propose = a*s_propose.T+ 0.5*s_propose*W*s_propose.T
        if (potential_s_propose[0,0]>potential_s[0,0]):
            s = s_propose
        else:
            u = np.random.rand(1)
            p = np.exp(potential_s_propose-potential_s)
            if(p[0,0]<u[0]):
                s = s_propose
        grad_a = grad_a+s
        grad_W = grad_W+s.T*s
    d['grad_a'] = grad_a
    d['grad_W'] = grad_W
    return d

def SA_grad_descent(S, a_init, W_init, stepsize, n_iter, sa_samples, burn_in_samples):
    a = a_init
    W = W_init
    n_obs = S.shape[0]
    grad_a = (1/float(n_obs))*(S.T*np.matrix(np.ones(n_obs)).T).T
    grad_W = (1/float(n_obs))*(S.T*S)
    for i in range(n_iter):
        d = SA_two_way_interaction(a, W, sa_samples, burn_in_samples)
        a = a + stepsize*(grad_a.T-(1/float(burn_in_samples))*d['grad_a'])
        W = W + stepsize*(grad_W-(1/float(burn_in_samples))*d['grad_W'])
    interactions = dict();
    interactions['scales'] = a
    interactions['weights'] = W
    return interactions