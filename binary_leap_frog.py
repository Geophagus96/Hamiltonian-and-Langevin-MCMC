# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 13:43:00 2020

@author: Yuze Zhou
"""

import numpy as np

def Leap_frog(a, A, W, D, x_init, p_init, burnin_samples, samples, leapfrogs, step_size):
    n_dim = W.shape[1]
    d = dict();
    A_inv = np.linalg.inv(A)
    WD_inv = np.linalg.inv((W+D))
    x = x_init
    p = p_init
    alpha_x = np.exp(A_inv*x+a-0.5*np.matrix(np.diag(D)).T)
    lx = np.multiply(alpha_x,1/(1+alpha_x))
    grad_U = A_inv.T*WD_inv*A_inv*x-A_inv*lx
    Energy = 0.5*x.T*A_inv.T*WD_inv*A_inv*x-np.sum(np.log(1+alpha_x))+0.5*np.sum(np.power(p))
    for i in range(burnin_samples):
        x_new = x
        p_new = p 
        lx_new = lx
        grad_new = grad_U
        alphax_new = alpha_x
        for j in range(leapfrogs):
            p_hat = p_new-(step_size/2)*grad_new
            x_new = x+step_size*p_hat
            alphax_new = np.exp(A_inv*x_new+a-0.5*np.matrix(np.diag(D)).T)
            lx_new = np.multiply(alphax_new,1/(1+alphax_new)) 
            grad_new = A_inv.T*WD_inv*A_inv*x_new-A_inv*lx_new
            p_new = p_hat-(step_size/2)*grad_new
        Energy_new = 0.5*x.T*A_inv.T*WD_inv*A_inv*x-np.sum(np.log(1+alpha_x))+0.5*np.sum(np.power(p))
        u = np.random.rand()
        accept_prob = np.exp(Energy-Energy_new)
        if u < accept_prob:
            x = x_new
            p = p_new
            alpha_x = alphax_new
            lx = lx_new
            grad_U = grad_new
            Energy = Energy_new
    s = np.matrix(np.zeros([n_dim, samples]))
    for i in range(samples):
        x_new = x
        p_new = p 
        lx_new = lx
        grad_new = grad_U
        alphax_new = alpha_x
        for j in range(leapfrogs):
            p_hat = p_new-(step_size/2)*grad_new
            x_new = x+step_size*p_hat
            alphax_new = np.exp(A_inv*x_new+a-0.5*np.matrix(np.diag(D)).T)
            lx_new = np.multiply(alphax_new,1/(1+alphax_new)) 
            grad_new = A_inv.T*WD_inv*A_inv*x_new-A_inv*lx_new
            p_new = p_hat-(step_size/2)*grad_new
        Energy_new = 0.5*x.T*A_inv.T*WD_inv*A_inv*x-np.sum(np.log(1+alpha_x))+0.5*np.sum(np.power(p))
        u = np.random.rand()
        accept_prob = np.exp(Energy-Energy_new)
        if u < accept_prob:
            x = x_new
            p = p_new
            alpha_x = alphax_new
            lx = lx_new
            grad_U = grad_new
            Energy = Energy_new
        su = np.random.rand(n_dim)
        s[:,i] = (su<lx)
        d["s"] = s
        return d
    
        
        
            
            
    
    