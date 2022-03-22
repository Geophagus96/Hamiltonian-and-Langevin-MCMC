# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 11:55:59 2021

@author: Yuze Zhou
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import trapz
import scipy.stats as st
from math import sqrt
import time
from copy import deepcopy    

def initialstate(N):   
    ''' generates a random spin configuration for initial condition'''
    state = 2*np.random.randint(2, size=(N,N))-1
    return state

def aux_initialize(N, p1):
    sign = 2*np.random.randint(2, size=(N,N))-1
    state = np.random.binomial(size=N*N,n=1,p=p1).reshape((N,N))
    return sign*state

#Energy calculation for a given configuration
def calcEnergy(config):
    energy = 0
    for i in range(len(config)):
        for j in range(len(config)):
            S = config[i,j]
            nb = config[(i+1)%N, j] + config[i,(j+1)%N] + config[(i-1)%N, j] + config[i,(j-1)%N]
            energy += -nb*S
    return energy/4.

#Energy calculation for a given configuration using matrix representation
neigh_matN = neigh_mat(N)
def calcEnergy_new(config):
    vec_config = np.reshape(config, N*N)
    return float((-0.25)*np.matrix(vec_config)*neigh_matN*np.matrix(vec_config).T)

#Forward Transition probability from 1 to 1
def Q11(p,alpha):
    if ((alpha+p)<=1):
        if (alpha<=p):
            return ((2*p-alpha)/(2*p))
        else:
            return (p/(2*alpha))
    else:
        if (alpha<=p):
            return  ((2*p-alpha)/(2*p))+(1/(alpha*p))*((alpha+p)**2/2-(alpha+p)+1/2)
        else:
            return (p/(2*alpha))+(1/(alpha*p))*((alpha+p)**2/2-(alpha+p)+1/2)

#Forward Transition probability from 0 to 1
def Q01(p,alpha):
    if (alpha<=(1-p)):
        if (alpha<=p):
            return (alpha/(2*(1-p)))
        else:
            return (1/(alpha*(1-p)))*(alpha*p-p**2/2)
    else:
        if (alpha<=p):
            return 1-(1-p)/(2*alpha)
        else:
            return 1-(1-p)/alpha+(1/(alpha*(1-p)))*(alpha*p-p**2/2-alpha*(alpha+p-1)+(alpha+p-1)**2/2)

#Magnetization calculation for a given configuration
def calcMag(config):
    mag = np.sum(config)
    return mag

def inv_operand(x, y, p):
    w = np.random.rand()
    if (x==1):
        if (y==0):
            u = -1
        else:
            if (w < ((p/2)/(1-p/2))):
                u = 1
            else:
                u = 0
    if (x==(-1)):
        if (y==1):
            u = 1
        else:
            if (w < ((p/2)/(1-p/2))):
                u = -1
            else:
                u = 0
    return u

#Thirteen dimensional with conditional independence neighbourhood with over-relaxation
def thirteen_noprod_cond(config, beta, alpha, Ene, Mag):
    a = np.random.randint(0,N)
    b = np.random.randint(0,N)
    row_candidates = np.array([a,a,a,(a+1)%N,(a+1)%N,(a+2)%N,(a+2)%N,(a+2)%N,(a+3)%N,(a+3)%N,(a+4)%N,(a+4)%N,(a+4)%N])
    col_candidates = np.array([b,(b+2)%N,(b+4)%N,(b+1)%N,(b+3)%N,b,(b+2)%N,(b+4)%N,(b+1)%N,(b+3)%N, b,(b+2)%N,(b+4)%N])
    for i in range(13):
        a_cand = row_candidates[i]
        b_cand = col_candidates[i]
        cand_nb = config[(a_cand+1)%N, b_cand] + config[a_cand,(b_cand+1)%N] + config[(a_cand-1)%N, b_cand] + config[a_cand,(b_cand-1)%N]
        p = np.exp(-beta*cand_nb)/(np.exp(-beta*cand_nb)+np.exp(beta*cand_nb))
        S = config[a_cand, b_cand]
        if (S==1):
            u = p*np.random.rand()
        else:
            u = (p+(1-p)*np.random.rand())
        u_new = (u+alpha*np.random.rand())%1
        if (u_new <= p):
            S_new = 1
        else:
            S_new = -1
        if (S!= S_new):
            Ene += (S*cand_nb)
            Mag += (2*S_new)
        config[a_cand, b_cand] = S_new
    return config, Ene, Mag

def thirteen_noprod_optim(config, beta, Ene, Mag):
    a = np.random.randint(0,N)
    b = np.random.randint(0,N)
    row_candidates = np.array([a,a,a,(a+1)%N,(a+1)%N,(a+2)%N,(a+2)%N,(a+2)%N,(a+3)%N,(a+3)%N,(a+4)%N,(a+4)%N,(a+4)%N])
    col_candidates = np.array([b,(b+2)%N,(b+4)%N,(b+1)%N,(b+3)%N,b,(b+2)%N,(b+4)%N,(b+1)%N,(b+3)%N, b,(b+2)%N,(b+4)%N])
    for i in range(13):
        a_cand = row_candidates[i]
        b_cand = col_candidates[i]
        cand_nb = config[(a_cand+1)%N, b_cand] + config[a_cand,(b_cand+1)%N] + config[(a_cand-1)%N, b_cand] + config[a_cand,(b_cand-1)%N]
        p = np.exp(-beta*cand_nb)/(np.exp(-beta*cand_nb)+np.exp(beta*cand_nb))
        S = config[a_cand, b_cand]
        if (S==1):
            u = p*np.random.rand()
        else:
            u = (p+(1-p)*np.random.rand())
        alpha = np.sqrt(p**2+(1-p)**2)
        u_new = (u+alpha*np.random.rand())%1
        if (u_new <= p):
            S_new = 1
        else:
            S_new = -1
        if (S!= S_new):
            Ene += (S*cand_nb)
            Mag += (2*S_new)
        config[a_cand, b_cand] = S_new
    return config, Ene, Mag

#Thirteen dimensional with conditional independence neighbourhood with Gibbs sampling
def thirteen_Gibbs(config, beta, Ene, Mag):
    a = np.random.randint(0,N)
    b = np.random.randint(0,N)
    row_candidates = np.array([a,a,a,(a+1)%N,(a+1)%N,(a+2)%N,(a+2)%N,(a+2)%N,(a+3)%N,(a+3)%N,(a+4)%N,(a+4)%N,(a+4)%N])
    col_candidates = np.array([b,(b+2)%N,(b+4)%N,(b+1)%N,(b+3)%N,b,(b+2)%N,(b+4)%N,(b+1)%N,(b+3)%N, b,(b+2)%N,(b+4)%N])
    for i in range(13):
        a_cand = row_candidates[i]
        b_cand = col_candidates[i]
        cand_nb = config[(a_cand+1)%N, b_cand] + config[a_cand,(b_cand+1)%N] + config[(a_cand-1)%N, b_cand] + config[a_cand,(b_cand-1)%N]
        E_cand = beta*cand_nb
        p_cand = np.exp(-E_cand)/(np.exp(-E_cand)+np.exp(E_cand))
        w = np.random.rand()
        x_cand = config[a_cand, b_cand]
        if (w<=p_cand):
            S_new = 1
        else:
            S_new = -1
        if (x_cand!= S_new):
            Ene += (x_cand*cand_nb)
            Mag += (2*S_new)
        config[a_cand, b_cand] = S_new
    return config, Ene, Mag

#Thirteen dimensional neighbourhood with conditional independence stucture with momemtum and over-relaxation
def thirteen_mom_cond(config, u, beta, alpha, p, Ene, Mag):
    a = np.random.randint(0,N)
    b = np.random.randint(0,N)
    row_candidates = np.array([a,a,a,(a+1)%N,(a+1)%N,(a+2)%N,(a+2)%N,(a+2)%N,(a+3)%N,(a+3)%N,(a+4)%N,(a+4)%N,(a+4)%N])
    col_candidates = np.array([b,(b+2)%N,(b+4)%N,(b+1)%N,(b+3)%N,b,(b+2)%N,(b+4)%N,(b+1)%N,(b+3)%N, b,(b+2)%N,(b+4)%N])
    for i in range(13):
        a_cand = row_candidates[i]
        b_cand = col_candidates[i]
        cand_nb = config[(a_cand+1)%N, b_cand] + config[a_cand,(b_cand+1)%N] + config[(a_cand-1)%N, b_cand] + config[a_cand,(b_cand-1)%N]
        x_cand = config[a_cand,b_cand]
        u_cand = -u[a_cand,b_cand]
        y = (x_cand+1)//2+u_cand
        if (y>0):
            y = 1
        else:
            y = 0
            
        p0 = np.exp(-beta*cand_nb)/(np.exp(-beta*cand_nb)+np.exp(beta*cand_nb))
        if (y==1):
            p1 = p0*(1-p/2)/(p0*(1-p/2)+(1-p0)*p/2)
        else:
            p1 = (p0*p/2)/(p0*p/2+(1-p0)*(1-p/2))
        if (x_cand==1):
            w = p1*np.random.rand()
        else:
            w = p1+(1-p1)*np.random.rand()
        w_new = (w+alpha*np.random.rand())%1
        if (w_new <= p1):
            S_new = 1
            u[a_cand, b_cand] = inv_operand(1, y, p)
        else:
            S_new = (-1)
            u[a_cand, b_cand] = inv_operand(-1, y, p)
        if (x_cand!=S_new):
             Ene += (x_cand*cand_nb)
             Mag += (2*S_new)
        config[a_cand, b_cand] = S_new
    return config, u, Ene, Mag

#Thirteen dimensional neighbourhood with conditional independence stucture with momemtum and over-relaxation
def thirteen_mom_optimal_alpha(config, u, beta, p, Ene, Mag):
    a = np.random.randint(0,N)
    b = np.random.randint(0,N)
    row_candidates = np.array([a,a,a,(a+1)%N,(a+1)%N,(a+2)%N,(a+2)%N,(a+2)%N,(a+3)%N,(a+3)%N,(a+4)%N,(a+4)%N,(a+4)%N])
    col_candidates = np.array([b,(b+2)%N,(b+4)%N,(b+1)%N,(b+3)%N,b,(b+2)%N,(b+4)%N,(b+1)%N,(b+3)%N, b,(b+2)%N,(b+4)%N])
    for i in range(13):
        a_cand = row_candidates[i]
        b_cand = col_candidates[i]
        cand_nb = config[(a_cand+1)%N, b_cand] + config[a_cand,(b_cand+1)%N] + config[(a_cand-1)%N, b_cand] + config[a_cand,(b_cand-1)%N]
        x_cand = config[a_cand,b_cand]
        u_cand = -u[a_cand,b_cand]
        y = (x_cand+1)//2+u_cand
        if (y>0):
            y = 1
        else:
            y = 0
        p0 = np.exp(-beta*cand_nb)/(np.exp(-beta*cand_nb)+np.exp(beta*cand_nb))
        if (y==1):
            p1 = p0*(1-p/2)/(p0*(1-p/2)+(1-p0)*p/2)
        else:
            p1 = (p0*p/2)/(p0*p/2+(1-p0)*(1-p/2))
        if (x_cand==1):
            w = p1*np.random.rand()
        else:
            w = p1+(1-p1)*np.random.rand()
        alpha = np.sqrt(p1**2+(1-p1)**2)
        w_new = (w+alpha*np.random.rand())%1
        if (w_new <= p1):
            S_new = 1
            u[a_cand, b_cand] = inv_operand(1, y, p)
        else:
            S_new = (-1)
            u[a_cand, b_cand] = inv_operand(-1, y, p)
        if (x_cand!=S_new):
             Ene += (x_cand*cand_nb)
             Mag += (2*S_new)
        config[a_cand, b_cand] = S_new
    return config, u, Ene, Mag
    

def neigh_mat(N):
    n_sites = N*N
    neigh_mat = np.zeros([n_sites, n_sites])
    for i in range(N):
        for j in range(N):
            candidate = i*N+j
            neighs1 = ((i+1)%N)*N+j
            neighs2 = ((i-1)%N)*N+j
            neighs3 = i*N+(j+1)%N
            neighs4 = i*N+(j-1)%N
            neigh_mat[candidate, neighs1] = 1
            neigh_mat[candidate, neighs2] = 1
            neigh_mat[candidate, neighs3] = 1
            neigh_mat[candidate, neighs4] = 1
    return neigh_mat


def global_momentum(config, u, beta, p, alpha, Ene):
    config_candidate = deepcopy(config)
    u_candidate = deepcopy(u)
    log_reject_ratio = 0
    for i in range(N):
        for j in range(N):
            cand_nb = config[(i+1)%N, j] + config[i,(j+1)%N] + config[(j-1)%N, j] + config[i,(j-1)%N] 
            x_cand = config[i,j] 
            u_cand = -u[i,j] 
            y = (x_cand+1)//2+u_cand
            if (y>0):
                y = 1
            else:
                y = 0
            p0 = np.exp(-beta*cand_nb)/(np.exp(-beta*cand_nb)+np.exp(beta*cand_nb)) 
            if (y==1):
                p1 = p0*(1-p/2)/(p0*(1-p/2)+(1-p0)*p/2) 
            else:
                p1 = (p0*p/2)/(p0*p/2+(1-p0)*(1-p/2))
            if (x_cand==1):
                w0 = p1*np.random.rand()
            else:
                w0 = p1+(1-p1)*np.random.rand()
            alpha = np.sqrt(p1**2+(1-p1)**2)
            w1 = (w0+alpha*np.random.rand())%1
            if (w1<=p1):
                x_new = 1
            else:
                x_new = -1
            config_candidate[i,j] = x_new
            u_new = inv_operand(x_new, y, p)
            u_candidate[i,j] = u_new
            if (x_cand!=x_new):
                if (x_cand == 1):
                    log_reject_ratio += (np.log(Q01(p1, alpha)) - np.log(1-Q11(p1,alpha)))
                else:
                    log_reject_ratio += (-np.log(Q01(p1, alpha))+np.log(1-Q11(p1,alpha)))
            if (y==0):
                if (x_cand == (-1)):
                    if (u_cand==(-1)):
                        log_reject_ratio -= np.log((p/2)/(1-p/2))
                    else:
                        log_reject_ratio -= np.log((1-p)/(1-p/2))
                if (x_new == (-1)):
                    if (u_new ==(-1)):
                        log_reject_ratio += np.log((p/2)/(1-p/2))
                    else:
                        log_reject_ratio += np.log((1-p)/(1-p/2))
            else:
                 if (x_cand == 1):
                    if (u_cand == 1):
                        log_reject_ratio -= np.log((p/2)/(1-p/2))
                    else:
                        log_reject_ratio -= np.log((1-p)/(1-p/2))
                 if (x_new == 1):
                    if (u_new == 1):
                        log_reject_ratio += np.log((p/2)/(1-p/2))
                    else:
                        log_reject_ratio += np.log((1-p)/(1-p))
    Ene_new = calcEnergy_new(config_candidate)
    log_reject_ratio += np.exp((-2*beta)*(Ene_new-Ene))
    w = np.random.rand()
    if (w<= min(1, float(log_reject_ratio))):
        Ene = Ene_new
        config = config_candidate
        u = u_candidate
    Mag = np.sum(config)
    return config, u, Ene, Mag

            
                
                    
                    
    
