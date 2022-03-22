# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 20:44:58 2022

@author: Yuze Zhou
"""

import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import acf
from scipy import trapz
import scipy.stats as st
from math import sqrt
from scipy.integrate import quad 
import collections

#Forward Transition probability from 1 to 1
def Q11(p,alpha):
    if ((alpha+p)<=1):
        if (alpha<=p):
            return ((2*p-alpha)/(2*p))
        else:
            return (p/(2*alpha))
    else:
        if (alpha<=p):
            return  ((2*p-alpha)/(2*p))+(1/(2*alpha*p))*(alpha+p-1)**2
        else:
            return (p/(2*alpha))+(1/(2*alpha*p))*(alpha+p-1)**2

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
            return 1-(1-p)/(2*alpha)+(1/(alpha*(1-p)))*(alpha*p-p**2/2-alpha*(alpha+p-1)+(alpha+p-1)**2/2)

#Backward Transition probability from 1 to 1    
def Qb11(p,alpha):
    if (alpha<=p):
        if ((alpha+p)<=1):
            return (2*p-alpha)/(2*p)
        else:
            return (alpha+p-1)**2/(2*alpha*p)+(2*p-alpha)/(2*p)
    else:
        if ((alpha+p)<=1):
            return p/(2*alpha)
        else:
            return p/(2*alpha)+(alpha+p-1)**2/(2*alpha*p)

#Backward Transition probability from 0 to 1
def Qb01(p,alpha):
    if(alpha<=(1-p)):
        if (alpha<=p):
            return alpha/(2*(1-p))
        else:
            return p*(2*alpha-p)/(2*alpha*(1-p))
    else:
        if (alpha<=p):
            return (2*alpha-1+p)/(2*alpha)
        else:
            return (1-alpha)*(1+alpha-2*p)/(2*alpha*(1-p))+(alpha+p-1)/alpha
        
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

def theoretical_val(beta):
    probs = np.zeros(8)
    interacts = 0.5*np.matrix(np.array([[0,1,1],[1,0,1],[1,1,0]]))
    for i in range(8):
        config = 2*np.array(list(bin(i)[2:].zfill(3))).astype(np.float)-1
        probs[i] = np.exp(-beta*np.matrix(config)*interacts*np.matrix(config).T)
    return probs/np.sum(probs)

def aux_initialize(p):
    sign = 2*np.random.randint(2, size=3)-1
    state = np.random.binomial(size=3,n=1,p=p)
    return sign*state

def three_dim_sampling(u, beta, p, n):
    x = 2*np.array(list(bin(n)[2:].zfill(3))).astype(np.float)-1
    x_cand = int((3*np.random.rand())//1)
    if (x_cand==0):
        px = np.exp(-beta*(x[1]+x[2]))/(np.exp(-beta*(x[1]+x[2]))+np.exp(beta*(x[1]+x[2])))
    elif (x_cand==1):
        px = np.exp(-beta*(x[0]+x[2]))/(np.exp(-beta*(x[0]+x[2]))+np.exp(beta*(x[0]+x[2])))
    else:
        px = np.exp(-beta*(x[0]+x[1]))/(np.exp(-beta*(x[0]+x[1]))+np.exp(beta*(x[0]+x[1])))
    x0 = int(x[x_cand])
    u0 = (-1)*u[x_cand]
    y = int(((x0+1)//2+u0)>0)
    if (y==1):
        pxy = px*(1-p/2)/(px*(1-p/2)+(1-px)*(p/2))
    else:
        pxy = px*(p/2)/(px*(p/2)+(1-px)*(1-p/2))
    if (x0==1):
        w0 = pxy*np.random.rand()
    else:
        w0 = pxy+(1-pxy)*np.random.rand()
    alpha= np.sqrt(pxy**2+(1-pxy)**2)
    w1 = (w0+alpha*np.random.rand())%1
    if (w1 <= pxy):
        x1 = 1
    else:
        x1 = -1
    u1 = inv_operand(x1, y ,p)
    x[x_cand] = x1
    u[x_cand] = u1
    x = -0.5*x+0.5
    x = x.astype('int')
    n = int(''.join(np.array(x).astype('str')), 2)
    return n, u

interacts = 0.5*np.matrix(np.array([[0,1,1],[1,0,1],[1,1,0]]))

def three_dim_sampling_2(u, beta, p, n):
    u_cand = -u
    x = 2*np.array(list(bin(n)[2:].zfill(3))).astype(np.float)-1
    x_cand = int((3*np.random.rand())//1)
    p1s = np.zeros(2)
    if (x_cand==0):
        x0 = np.array([x[1],x[2]])
        u0 = np.array([u_cand[1], u_cand[2]])
        p1s[0] = np.exp(-beta*(x[0]+x[2]))/(np.exp(-beta*(x[0]+x[2]))+np.exp(beta*(x[0]+x[2])))
        p1s[1] = np.exp(-beta*(x[0]+x[1]))/(np.exp(-beta*(x[0]+x[1]))+np.exp(beta*(x[0]+x[1])))
    elif (x_cand==1):
        x0 = np.array([x[0],x[2]])
        u0 = np.array([u_cand[0],u_cand[2]])
        p1s[0] = np.exp(-beta*(x[1]+x[2]))/(np.exp(-beta*(x[1]+x[2]))+np.exp(beta*(x[1]+x[2])))
        p1s[1] = np.exp(-beta*(x[0]+x[1]))/(np.exp(-beta*(x[0]+x[1]))+np.exp(beta*(x[0]+x[1])))
    else:
        x0 = np.array([x[0],x[1]])
        u0 = np.array([u_cand[0],u_cand[1]])
        p1s[0] = np.exp(-beta*(x[1]+x[2]))/(np.exp(-beta*(x[1]+x[2]))+np.exp(beta*(x[1]+x[2])))
        p1s[1] = np.exp(-beta*(x[0]+x[2]))/(np.exp(-beta*(x[0]+x[2]))+np.exp(beta*(x[0]+x[2])))
    y = (x0+1)//2+u0 
    y[y>0] = 1 
    y[y<=0] = 0
    xnew = np.zeros(2)
    unew = np.zeros(2)
    pys = np.zeros(2)
    for i in range(2):
        if (y[i]==1):
            pys[i] = (p1s[i]*(1-p/2))/(p1s[i]*(1-p/2)+(1-p1s[i])*(p/2))
        else:
            pys[i] = (p1s[i]*(p/2))/(p1s[i]*(p/2)+(1-p1s[i])*(1-p/2))
    alphas = np.zeros(2)
    for i in range(2):
       if (x0[i]==1):
           w0 = pys[i]*np.random.rand()  
       else:
           w0 = pys[i]+(1-pys[i])*np.random.rand()
       alphas[i] = np.sqrt(pys[i]**2+(1-pys[i])**2)
       w = (w0+alphas[i]*np.random.rand())%1
       if(w<=pys[i]):
           xnew[i] = 1 
       else:
           xnew[i] = -1
       unew[i] = inv_operand(xnew[i], y[i], p)            
       
    log_reject_ratio = 0
    for i in range(2):
        if (y[i] == 1):
            if (x0[i]!=xnew[i]):
                log_reject_ratio += (x0[i])*np.log((Q01(pys[i],alphas[i])*(p/2))/((1-Q11(pys[i],alphas[i]))*(1-p/2)))
                
            
            
        else:
            if (x0[i]!=xnew[i]):
                log_reject_ratio += (x0[i])*np.log((Q01(pys[i],alphas[i])*(1-p/2))/((1-Q11(pys[i],alphas[i]))*(p/2)))
    x1 = x
    u1 = u_cand
    if (x_cand ==0):
        x1[1] = xnew[0]
        x1[2] = xnew[1]
        u1[1] = unew[0]
        u1[2] = unew[1]
    elif (x_cand == 1):
        x1[0] = xnew[0]
        x1[2] = xnew[1]
        u1[0] = unew[0]
        u1[2] = unew[1]
    else:
        x1[0] = xnew[0]
        x1[1] = xnew[1]
        u1[0] = xnew[0]
        u1[1] = xnew[1]
    reject_ratio = np.exp(log_reject_ratio-beta*(np.matrix(x1)*interacts*np.matrix(x1).T-np.matrix(x)*interacts*np.matrix(x).T))
    if (np.random.rand()<reject_ratio):
        x = x1
        u = u1
        acc = 1
    else:
        acc = 0
        u = u_cand
    x = -0.5*x+0.5
    x = x.astype('int')
    n = int(''.join(np.array(x).astype('str')), 2)
    return n, u, acc


def three_dim_sampling_3(u, beta, p, n):
    u_cand = -u
    x = 2*np.array(list(bin(n)[2:].zfill(3))).astype(np.float)-1
    x_cand = int((3*np.random.rand())//1)
    p1s = np.zeros(2)
    if (x_cand==0):
        x0 = np.array([x[1],x[2]])
        u0 = np.array([u_cand[1], u_cand[2]])
        p1s[0] = np.exp(-beta*(x[0]+x[2]))/(np.exp(-beta*(x[0]+x[2]))+np.exp(beta*(x[0]+x[2])))
        p1s[1] = np.exp(-beta*(x[0]+x[1]))/(np.exp(-beta*(x[0]+x[1]))+np.exp(beta*(x[0]+x[1])))
    elif (x_cand==1):
        x0 = np.array([x[0],x[2]])
        u0 = np.array([u_cand[0],u_cand[2]])
        p1s[0] = np.exp(-beta*(x[1]+x[2]))/(np.exp(-beta*(x[1]+x[2]))+np.exp(beta*(x[1]+x[2])))
        p1s[1] = np.exp(-beta*(x[0]+x[1]))/(np.exp(-beta*(x[0]+x[1]))+np.exp(beta*(x[0]+x[1])))
    else:
        x0 = np.array([x[0],x[1]])
        u0 = np.array([u_cand[0],u_cand[1]])
        p1s[0] = np.exp(-beta*(x[1]+x[2]))/(np.exp(-beta*(x[1]+x[2]))+np.exp(beta*(x[1]+x[2])))
        p1s[1] = np.exp(-beta*(x[0]+x[2]))/(np.exp(-beta*(x[0]+x[2]))+np.exp(beta*(x[0]+x[2])))
    y = (x0+1)//2+u0 
    y[y>0] = 1 
    y[y<=0] = 0
    xnew = np.zeros(2)
    unew = np.zeros(2)
    pys = np.zeros(2)
    alphas = np.zeros(2)
    qs = np.zeros(2)
    for i in range(2):
        if (y[i]==1):
            pys[i] = (p1s[i]*(1-p/2))/(p1s[i]*(1-p/2)+(1-p1s[i])*(p/2))
            alphas[i] = np.sqrt(pys[i]**2+(1-pys[i])**2)
            qs[i] = Q11(pys[i], alphas[i])
        else:
            pys[i] = (p1s[i]*(p/2))/(p1s[i]*(p/2)+(1-p1s[i])*(1-p/2))
            alphas[i] = np.sqrt(pys[i]**2+(1-pys[i])**2)
            qs[i] = pys[i]*(1-Q11(pys[i],alphas[i]))/(1-pys[i])
    for i in range(2):
        if (np.random.rand() < qs[i]):
            xnew[i] = 1
        else:
            xnew[i] = (-1)
        unew[i] = inv_operand(xnew[i], y[i], p)
    log_reject_ratio = 0
    for i in range(2):
        if (y[i] == 1):
            if (x0[i]!=xnew[i]):
                log_reject_ratio += (x0[i])*np.log((pys[i]*(1-qs[i])/(1-pys[i])*(p/2))/((1-qs[i])*(1-p/2)))
                
            
            
        else:
            if (x0[i]!=xnew[i]):
                log_reject_ratio += (x0[i])*np.log((qs[i]*(1-p/2))/((qs[i]*(1-pys[i])/pys[i])*(p/2)))
    x1 = x
    u1 = u_cand
    if (x_cand ==0):
        x1[1] = xnew[0]
        x1[2] = xnew[1]
        u1[1] = unew[0]
        u1[2] = unew[1]
    elif (x_cand == 1):
        x1[0] = xnew[0]
        x1[2] = xnew[1]
        u1[0] = unew[0]
        u1[2] = unew[1]
    else:
        x1[0] = xnew[0]
        x1[1] = xnew[1]
        u1[0] = xnew[0]
        u1[1] = xnew[1]
    reject_ratio = np.exp(log_reject_ratio-beta*(np.matrix(x1)*interacts*np.matrix(x1).T-np.matrix(x)*interacts*np.matrix(x).T))
    if (np.random.rand()<reject_ratio):
        x = x1
        u = u1
        acc = 1
    else:
        acc = 0
        u = u_cand
    x = -0.5*x+0.5
    x = x.astype('int')
    n = int(''.join(np.array(x).astype('str')), 2)
    return n, u, acc


q = 0.7
ns = np.zeros(160000)
u = aux_initialize(q)
beta = np.log(6)/4
theo_prob = theoretical_val(beta)
acc_num = 0
n = 3
for i in range(80000):
    n, u, acc = three_dim_sampling_3(u, beta, q, n)

for i in range(160000):
    n, u, acc = three_dim_sampling_3(u, beta, q, n)
    ns[i] = n
    acc_num += acc

print(theo_prob)
print(acc_num/160000)
print(np.array(list(collections.Counter(ns).values()))/160000)


    
    
 