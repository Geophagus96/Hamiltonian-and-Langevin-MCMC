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
        u_cand = u[a_cand,b_cand]
        y = 0.5*x_cand+x_cand+u_cand
        y = y-(y>1)+(y<0)
        y = 2*y-1     
        y = y.astype(int)
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


