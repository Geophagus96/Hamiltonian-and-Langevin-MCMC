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

#Given x1 and y, inv-operand is the function to generate u1 such that x1+u1=y
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

def aux_initialize(p, n0):
    sign = 2*np.random.randint(2, size =n0)-1
    state = np.random.binomial(size=n0,n=1,p=p)
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

def two_dim_sampling(u, beta, p ,n):
    u0 = u
    x0 = 2*np.array(list(bin(n)[2:].zfill(2))).astype(np.float)-1
    p1s = 0.5*np.ones(2)
    y = (x0+1)//2+u0 
    y[y>0] = 1 
    y[y<=0] = 0
    pys = np.zeros(2)
    x1 = np.zeros(2)
    u1 = np.zeros(2)
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
           x1[i] = 1 
       else:
           x1[i] = -1
       u1[i] = inv_operand(x1[i], y[i], p)
    rej_ratio = 1
    if (x1[0]!=x1[1]):
        rej_ratio *= np.exp(beta)
    else:
        rej_ratio *= np.exp(-beta)
    if (x0[0]!=x1[1]):
        rej_ratio *= np.exp(-beta)
    else:
        rej_ratio *= np.exp(-beta)
    if (np.random.rand()<=rej_ratio):
        x = x1
        u = u1
        acc = 1
    else:
        x = x0
        u = u0
        acc = 0
    x = -0.5*x+0.5
    x = x.astype('int')
    n = int(''.join(np.array(x).astype('str')), 2)
    return n, u, acc

def two_dim_sampling_1(u, beta, k, p ,n):
    u0 = u
    x0 = 2*np.array(list(bin(n)[2:].zfill(2))).astype(np.float)-1
    #p1s is the list containing marginal distribution, the first element is
    #p(x_1=1) while the second is p(x_2=1)
    p1s = np.zeros(2)
    p1s[0] = np.exp(-k)/(np.exp(-k)+np.exp(k))
    p1s[1] = (np.exp(beta+k)+np.exp(-beta-k))/((np.exp(beta)+np.exp(-beta))*(np.exp(k)+np.exp(-k)))
    #Computing y as the product of x and u
    y = (x0+1)//2+u0 
    y[y>0] = 1 
    y[y<=0] = 0
    qs = np.zeros(2)
    x1 = np.zeros(2)
    u1 = np.zeros(2)
    #Calculating the conditional distributions of p(x_1=1|y_1) and p(x_2=1|y_2)
    for i in range(2): 
        if (y[i]==1):
            qs[i] = (p1s[i]*(1-p/2))/(p1s[i]*(1-p/2)+(1-p1s[i])*(p/2))
        else:
            qs[i] = ((1-p1s[i])*(1-p/2))/((1-p1s[i])*(1-p/2)+p1s[i]*(p/2))
    #Flipping using the momentum over-relaxation
    for i in range(2):
        if (y[i]==1):
            if (np.random.rand()<=qs[i]):
                x1[i] = 1
                if (np.random.rand()<=((1-p)/(1-p/2))):
                    u1[i] = 0
                else:
                    u1[i] = 1
            else:
                x1[i] = -1
                u1[i] = 1
        else: 
            if (np.random.rand()<=qs[i]):
                x1[i] = -1
                if (np.random.rand()<=((1-p)/(1-p/2))):
                    u1[i] = 0
                else:
                    u1[i] = -1
            else:
                x1[i] = 1
                u1[i] = -1
    rej_ratio = 1
    pb = 1
    pf = 1
    #Calculating the forward transition probability, which is the product of
    #marginal distributions for x1
    pb *= np.exp(-k*x1[0])
    pb *= (np.exp(-beta-k*x1[1])+np.exp(beta+k*x1[1]))
    #Calculating the backward transition probability, which is the product of
    #marginal distributiosn for x0
    pf *= np.exp(-k*x0[0])
    pf *= (np.exp(-beta-k*x0[1])+np.exp(beta+k*x0[1]))
    #Calculating the accpetance probability
    rej_ratio = np.exp(-beta*x1[0]*x1[1]-k*x1[0]+beta*x0[0]*x0[1]+k*x0[0])*pf/pb
    if (np.random.rand()<=rej_ratio):
        x = x1
        u = u1
        acc = 1
    else:
        x = x0
        u = u0
        acc = 0
    x = -0.5*x+0.5
    x = x.astype('int')
    n = int(''.join(np.array(x).astype('str')), 2)
    return n, u, acc



#Momentum Product Approximation for two-dimensional toy example with
#p(x_1, x_2) = exp(-beta*x_1*x_2-k*x_1)
#Notations in the codes, x_1, x_2 are the variables for the target distribution
#While x0 is the original sample, x1 is the new proposal

def two_dim_sampling_2(u, beta, k, p ,n):
    u0 = u
    x0 = 2*np.array(list(bin(n)[2:].zfill(2))).astype(np.float)-1
    #p1s is the list containing marginal distribution, the first element is
    #p(x_1=1) while the second is p(x_2=1)
    p1s = np.zeros(2)
    p1s[0] = np.exp(-k)/(np.exp(-k)+np.exp(k))
    p1s[1] = (np.exp(beta+k)+np.exp(-beta-k))/((np.exp(beta)+np.exp(-beta))*(np.exp(k)+np.exp(-k)))
    #Computing y as the product of x and u
    y = (x0+1)//2+u0 
    y[y>0] = 1 
    y[y<=0] = 0
    pys = np.zeros(2)
    x1 = np.zeros(2)
    u1 = np.zeros(2)
    #Calculating the conditional distributions of p(x_1=1|y_1) and p(x_2=1|y_2)
    for i in range(2): 
        if (y[i]==1):
            pys[i] = (p1s[i]*(1-p/2))/(p1s[i]*(1-p/2)+(1-p1s[i])*(p/2))
        else:
            pys[i] = (p1s[i]*(p/2))/(p1s[i]*(p/2)+(1-p1s[i])*(1-p/2))
    alphas = np.zeros(2)
    #Flipping using the momentum over-relaxation
    for i in range(2):
       if (x0[i]==1):
           w0 = pys[i]*np.random.rand()  
       else:
           w0 = pys[i]+(1-pys[i])*np.random.rand()
       alphas[i] = np.sqrt(pys[i]**2+(1-pys[i])**2)
       w = (w0+alphas[i]*np.random.rand())%1
       if(w<=pys[i]):
           x1[i] = 1 
       else:
           x1[i] = -1
    #Inv_operand is the function to get u1 after x1 is sampled satidfing
    #the product x1+u1=y
       u1[i] = inv_operand(x1[i], y[i], p)
    rej_ratio = 1
    pb = 1
    pf = 1
    #Calculating the forward transition probability, which is the product of
    #marginal distributions for x1
    pb *= np.exp(-k*x1[0])
    pb *= (np.exp(-beta-k*x1[1])+np.exp(beta+k*x1[1]))
    #Calculating the backward transition probability, which is the product of
    #marginal distributiosn for x0
    pf *= np.exp(-k*x0[0])
    pf *= (np.exp(-beta-k*x0[1])+np.exp(beta+k*x0[1]))
    #Calculating the accpetance probability
    rej_ratio = np.exp(-beta*x1[0]*x1[1]-k*x1[0]+beta*x0[0]*x0[1]+k*x0[0])*pf/pb
    if (np.random.rand()<=rej_ratio):
        x = x1
        u = u1
        acc = 1
    else:
        x = x0
        u = u0
        acc = 0
    x = -0.5*x+0.5
    x = x.astype('int')
    n = int(''.join(np.array(x).astype('str')), 2)
    return n, u, acc


def two_dim_sampling_3(u, beta, k, p, n, q1, q0):
    u0 = u
    x0 = 2*np.array(list(bin(n)[2:].zfill(2))).astype(np.float)-1
    y = (x0+1)//2+u0 
    y[y>0] = 1 
    y[y<=0] = 0
    x1 = np.zeros(2)
    u1 = np.zeros(2)
    p1s = np.zeros(2)
    p1s[0] = np.exp(-k)/(np.exp(-k)+np.exp(k))
    p1s[1] = (np.exp(beta+k)+np.exp(-beta-k))/((np.exp(beta)+np.exp(-beta))*(np.exp(k)+np.exp(-k)))
    for i in range(2):
        if (y[i] == 1):
            a = (p1s[i]*(1-p/2))/(p1s[i]*(1-p/2)+(1-p1s[i])*(p/2))
            if (x0[i] == 1):
                w1 = np.random.rand()
                if (w1 <= q1):
                    x1[i] = 1
                    w2 = np.random.rand()
                    if (w2 <= ((1-p)/(1-p/2))):
                        u1[i] = 0
                    else:
                        u1[i] = 1
                else:
                    x1[i] = -1
                    u1[i] = 1
            else:
                w1 = np.random.rand()
                if (w1 <= (a*(1-q1)/(1-a))):
                    x1[i] = 1
                    w2 = np.random.rand()
                    if (w2 <= ((1-p)/(1-p/2))):
                        u1[i] = 0
                    else:
                        u1[i] = 1
                else:
                    x1[i] = -1
                    u1[i] = 1
        else:
            b =  ((1-p1s[i])*(1-p/2))/((1-p1s[i])*(1-p/2)+p1s[i]*(p/2))
            if (x0[i] != 1):
                w1 = np.random.rand()
                if (w1 <= q0):
                    x1[i] = -1
                    w2 = np.random.rand()
                    if (w2 <= ((1-p)/(1-p/2))):
                        u1[i] = 0
                    else:
                        u1[i] = -1
                else:
                    x1[i] = 1
                    u1[i] = -1
            else:
                w1 = np.random.rand()
                if (w1 <= (b*(1-q0)/(1-b))):
                    x1[i] = -1
                    w2 = np.random.rand()
                    if (w2 <= ((1-p)/(1-p/2))):
                        u1[i] = 0
                    else:
                        u1[i] = 1
                else:
                    x1[i] = 1
                    u1[i] = -1
            
            
    rej_ratio = 1
    pb = 1
    pf = 1
    pb *= np.exp(-k*x1[0])
    pb *= (np.exp(-beta-k*x1[1])+np.exp(beta+k*x1[1]))
    pf *= np.exp(-k*x0[0])
    pf *= (np.exp(-beta-k*x0[1])+np.exp(beta+k*x0[1]))
    
    rej_ratio = np.exp(-beta*x1[0]*x1[1]-k*x1[0]+beta*x0[0]*x0[1]+k*x0[0])*pb/pf
    if (np.random.rand()<=rej_ratio):
        x = x1
        u = u1
        acc = 1
    else:
        x = x0
        u = u0
        acc = 0
    x = -0.5*x+0.5
    x = x.astype('int')
    n = int(''.join(np.array(x).astype('str')), 2)
    return n, u, acc


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

#Gibbs sampling for three dimensional toy example
def three_dim_Gibbs(beta, n):
    x = 2*np.array(list(bin(n)[2:].zfill(3))).astype(np.float)-1
    x_cand = int((3*np.random.rand())//1)
    x0 = np.zeros(2)
    #Randomly Select two variables to do sampling with product approximation,
    #Assume the two variables being selected are (x_1, x_2), the four 
    #probabilities are each, p11 = p(x_1=1,x_2=1), p10 = p(x_1=1, x_2=-1)
    #p01 = p(x_1=-1, x_2=1), p00 = p(x_1=-1, x_2=-1)
    if (x_cand == 0):
        x0[0] = x[1]
        x0[1] = x[2]
        p11 = np.exp(-beta*(1+2*x[0]))
        p10 = np.exp(beta)
        p00 = np.exp(-beta*(1-2*x[0]))
    elif (x_cand == 1):
        x0[0] = x[0]
        x0[1] = x[2]
        p11 = np.exp(-beta*(1+2*x[1]))
        p10 = np.exp(beta)
        p00 = np.exp(-beta*(1-2*x[1]))
    else:
        x0[0] = x[0]
        x0[1] = x[1]
        p11 = np.exp(-beta*(1+2*x[2]))
        p10 = np.exp(beta)
        p00 = np.exp(-beta*(1-2*x[2]))
    psum = (p11+2*p10+p00)
    x1 = np.zeros(2)
    w = np.random.rand()
    #Select the four candidates, (1,1), (1,-1), (-1,1), (-1,-1) with exact 
    #probability p11, p10, p01, p00
    if (w<= (p11/psum)):
        x1[0] = 1
        x1[1] = 1
    elif (w>=((p11+2*p10)/psum)):
        x1[0] = -1
        x1[1] = -1
    else:
        if (np.random.rand()<=0.5):
            x1[0] = 1
            x1[1] = -1
        else:
            x1[0] = -1
            x1[1] = 1
    if (x_cand == 0):
        x[1] = x1[0]
        x[2] = x1[1]
    elif (x_cand == 1):
        x[0] = x1[0]
        x[2] = x1[1]
    else:
        x[0] = x1[0]
        x[1] = x1[1]                    
    x = -0.5*x+0.5
    x = x.astype('int')
    n = int(''.join(np.array(x).astype('str')), 2)
    return n
    
        
            
def three_dim_sampling_3(u, beta, p, n):
    x = 2*np.array(list(bin(n)[2:].zfill(3))).astype(np.float)-1
    x_cand = int((3*np.random.rand())//1)
    x0 = np.zeros(2)
    u0 = np.zeros(2)
    if (x_cand == 0):
        x0[0] = x[1]
        x0[1] = x[2]
        u0[0] = u[1]
        u0[1] = u[2]
        p11 = np.exp(-beta*(1+2*x[0]))
        p10 = np.exp(beta)
        p00 = np.exp(-beta*(1-2*x[0]))
        fix = x[0]
    elif (x_cand == 1):
        x0[0] = x[0]
        x0[1] = x[2]
        u0[0] = u[0]
        u0[1] = u[2]
        p11 = np.exp(-beta*(1+2*x[1]))
        p10 = np.exp(beta)
        p00 = np.exp(-beta*(1-2*x[1]))
        fix = x[1]
    else:
        x0[0] = x[0]
        x0[1] = x[1]
        u0[0] = u[0]
        u0[1] = u[1]
        p11 = np.exp(-beta*(1+2*x[2]))
        p10 = np.exp(beta)
        p00 = np.exp(-beta*(1-2*x[2]))
        fix = x[2]
    p1s = (p11+p10)/(p11+2*p10+p00)
    y = (x0+1)//2+u0
    y[y>0] = 1
    y[y<=0] = 0
    x1 = np.zeros(2)
    u1 = np.zeros(2)
    for i in range(2):
        if (y[i]==1):
            a = (p1s*(1-p/2))/(p1s*(1-p/2)+(1-p1s)*(p/2))
            alpha = np.sqrt(a**2+(1-a)**2)
            if (x0[i]==1):
                w1 = (a*np.random.rand()+alpha*np.random.rand())%1
                if (w1 <= a):
                    x1[i] = 1
                    if (np.random.rand()<((1-p)/(1-p/2))):
                        u1[i] = 0
                    else:
                        u1[i] = 1
                else:
                    x1[i] = -1
                    u1[i] = 1
            else:
                w1 = (a+(1-a)*np.random.rand()+alpha*np.random.rand())%1
                if (w1 <= a):
                    x1[i] = 1
                    if (np.random.rand()<((1-p)/(1-p/2))):
                        u1[i] = 0
                    else:
                        u1[i] = 1
                else:
                    x1[i] = -1
                    u1[i] = 1
        else:
            a = (p1s*(p/2))/((1-p1s)*(1-p/2)+p1s*(p/2))
            alpha = np.sqrt(a**2+(1-a)**2)
            if (x0[i] == 1):
                w1 = (a*np.random.rand()+alpha*np.random.rand())%1
                if (w1 > a):
                    x1[i] = -1
                    if (np.random.rand()<((1-p)/(1-p/2))):
                        u1[i] = 0
                    else:
                        u1[i] = -1
                else:
                    x1[i] = 1
                    u1[i] = -1
            else:
                w1 = (a+(1-a)*np.random.rand()+alpha*np.random.rand())%1
                if (w1 >= a):
                    x1[i] = -1
                    if (np.random.rand()<((1-p)/(1-p/2))):
                        u1[i] = 0
                    else:
                        u1[i] = -1
                else:
                    x1[i] = 1
                    u1[i] = -1
    rej_ratio = 1.0
    for i in range(2):
        if (x0[i] == 1):
            rej_ratio *= (p11+p10)
        else:
            rej_ratio *= (p10+p00)
        if (x1[i] == 1):
            rej_ratio *= (1/(p11+p10))
        else:
            rej_ratio *= (1/(p10+p00))
    rej_ratio *= np.exp(-beta*(x1[0]*x1[1]+fix*(x1[0]+x1[1]))+beta*(x0[0]*x0[1]+fix*(x0[0]+x0[1])))
    xstar = x
    ustar = u
    if (x_cand == 0):
        xstar[1] = x1[0]
        xstar[2] = x1[1]
        ustar[1] = u1[0]
        ustar[2] = u1[1]
    elif (x_cand == 1):
        xstar[0] = x1[0]
        xstar[2] = x1[1]
        ustar[0] = u1[0]
        ustar[2] = u1[1]
    else:
        xstar[0] = x1[0]
        xstar[1] = x1[1]
        ustar[0] = u1[0]
        ustar[1] = u1[1]
    if (np.random.rand()<=rej_ratio):
        x = xstar
        u = ustar
        acc = 1
    else:
        acc = 0
                    
    x = -0.5*x+0.5
    x = x.astype('int')
    n = int(''.join(np.array(x).astype('str')), 2)
    return n, u, acc

def three_dim_sampling_4(u, beta, p, n):
    u_cand = -u
    x = 2*np.array(list(bin(n)[2:].zfill(3))).astype(np.float)-1
    x_cand = int((3*np.random.rand())//1)
    p1f = np.zeros(2)
    if (x_cand==0):
        x0 = np.array([x[1],x[2]])
        u0 = np.array([u_cand[1], u_cand[2]])
        p1f[0] = np.exp(-beta*(x[0]+x[2]))/(np.exp(-beta*(x[0]+x[2]))+np.exp(beta*(x[0]+x[2])))
        p1f[1] = np.exp(-beta*(x[0]+x[1]))/(np.exp(-beta*(x[0]+x[1]))+np.exp(beta*(x[0]+x[1])))
    elif (x_cand==1):
        x0 = np.array([x[0],x[2]])
        u0 = np.array([u_cand[0],u_cand[2]])
        p1f[0] = np.exp(-beta*(x[1]+x[2]))/(np.exp(-beta*(x[1]+x[2]))+np.exp(beta*(x[1]+x[2])))
        p1f[1] = np.exp(-beta*(x[0]+x[1]))/(np.exp(-beta*(x[0]+x[1]))+np.exp(beta*(x[0]+x[1])))
    else:
        x0 = np.array([x[0],x[1]])
        u0 = np.array([u_cand[0],u_cand[1]])
        p1f[0] = np.exp(-beta*(x[1]+x[2]))/(np.exp(-beta*(x[1]+x[2]))+np.exp(beta*(x[1]+x[2])))
        p1f[1] = np.exp(-beta*(x[0]+x[2]))/(np.exp(-beta*(x[0]+x[2]))+np.exp(beta*(x[0]+x[2])))
    y = (x0+1)//2+u0 
    y[y>0] = 1 
    y[y<=0] = 0
    x1 = np.zeros(2)
    u1 = np.zeros(2)
    qf = 1
    for i in range(2):
        if (y[i] == 1):
            a = (p1f[i]*(1-p/2))/(p1f[i]*(1-p/2)+(1-p1f[i])*(p/2))
            q = Q11(p, np.sqrt(a**2+(1-a)**2))
            if (x0[i]==1):
                if (np.random.rand()<=q):
                    qf *= q
                    x1[i] = 1
                    if (np.random.rand()<=((1-p)/(1-p/2))):
                        u1[i] = 0
                        qf *= ((1-p)/(1-p/2))
                    else:
                        u1[i] = 1
                        qf *= ((p/2)/(1-p/2))

                else:
                    qf *= (1-q)
                    x1[i] = -1
                    u1[i] = 1
            else:
                if (np.random.rand()<=(a*(1-q)/(1-a))):
                    qf *= (a*(1-q)/(1-a))
                    x1[i] = 1
                    if (np.random.rand()<=((1-p)/(1-p/2))):
                        qf *= ((1-p)/(1-p/2))
                        u1[i] = 0
                    else:
                        qf *= ((p/2)/(1-p/2))
                        u1[i] = 1
                else:
                    qf *= (1-a*(1-q)/(1-a))
                    x1[i] = -1
                    u1[i] = 1
        else:
            b = ((1-p1f[i])*(1-p/2))/((1-p1f[i])*(1-p/2)+p1f[i]*(p/2))
            a = 1-b
            q = 1-a*Q11(a,np.sqrt(a**2+(1-a)**2))/(1-a)
            if (x0[i] != 1):
                if (np.random.rand()<=q):
                    qf *= q
                    x1[i] = -1
                    if (np.random.rand()<=((1-p)/(1-p/2))):
                        qf *= ((1-p)/(1-p/2))
                        u1[i] = 0
                    else:
                        qf *= ((p/2)/(1-p/2))
                        u1[i] = -1
                else:
                    qf *= (1-q)
                    x1[i] = 1
                    u1[i] = -1
            else:
                if (np.random.rand()<=(b*(1-q)/(1-b))):
                    qf *= (b*(1-q)/(1-b))
                    x1[i] = -1
                    if (np.random.rand()<=((1-p)/(1-p/2))):
                        qf *= ((1-p)/(1-p/2))
                        u1[i] = 0
                    else:
                        qf *= ((p/2)/(1-p/2))
                        u1[i] = -1
                else:
                    qf *=  (1-b*(1-q)/(1-b))
                    x1[i] = 1
                    u1[i] = -1
    xstar = x
    ustar = u_cand
    if (x_cand==0):
        xstar[1] = x1[0]
        xstar[2] = x1[1]
        ustar[1] = u1[0]
        ustar[2] = u1[1]
    elif (x_cand==1):
        xstar[0] = x1[0]
        xstar[2] = x1[1]
        ustar[0] = u1[0]
        ustar[2] = u1[1]
    else:
        xstar[0] = x1[0]
        xstar[1] = x1[1]
        ustar[0] = u1[0]
        ustar[1] = u1[1]
    
    p1b = np.zeros(2)
    if (x_cand==0):
        x0 = np.array([x[1],x[2]])
        u0 = np.array([u_cand[1], u_cand[2]])
        p1b[0] = np.exp(-beta*(xstar[0]+xstar[2]))/(np.exp(-beta*(xstar[0]+xstar[2]))+np.exp(beta*(xstar[0]+xstar[2])))
        p1b[1] = np.exp(-beta*(xstar[0]+xstar[1]))/(np.exp(-beta*(xstar[0]+xstar[1]))+np.exp(beta*(xstar[0]+xstar[1])))
    elif (x_cand==1):
        x0 = np.array([x[0],x[2]])
        u0 = np.array([u_cand[0],u_cand[2]])
        p1b[0] = np.exp(-beta*(xstar[1]+xstar[2]))/(np.exp(-beta*(xstar[1]+xstar[2]))+np.exp(beta*(xstar[1]+xstar[2])))
        p1b[1] = np.exp(-beta*(xstar[0]+xstar[1]))/(np.exp(-beta*(xstar[0]+xstar[1]))+np.exp(beta*(xstar[0]+xstar[1])))
    else:
        x0 = np.array([x[0],x[1]])
        u0 = np.array([u_cand[0],u_cand[1]])
        p1b[0] = np.exp(-beta*(xstar[1]+xstar[2]))/(np.exp(-beta*(xstar[1]+xstar[2]))+np.exp(beta*(xstar[1]+xstar[2])))
        p1b[1] = np.exp(-beta*(xstar[0]+xstar[2]))/(np.exp(-beta*(xstar[0]+xstar[2]))+np.exp(beta*(xstar[0]+xstar[2])))
    qb = 1
    for i in range(2):
        if (y[i]==1):
            a = (p1b[i]*(1-p/2))/(p1b[i]*(1-p/2)+(1-p1b[i])*(p/2))
            q = Q11(p, np.sqrt(a**2+(1-a)**2))
            if (x1[i] == 1):
                if (x0[i]==1):
                    qb *= q
                    if (u0[i] == 0):
                        qb *= ((1-p)/(1-p/2))
                    else:
                        qb *= ((p/2)/(1-p/2))
                else:
                    qb *= (1-q)
            else:
                if (x0[i] == 1):
                    qb *= (a*(1-q)/(1-a))
                    if (u0[i] == 0):
                        qb *= ((1-p)/(1-p/2))
                    else:
                        qb *= ((p/2)/(1-p/2))
                else:
                    qb *= (1-a*(1-q)/(1-a))
        else:
            b = ((1-p1b[i])*(1-p/2))/((1-p1b[i])*(1-p/2)+p1b[i]*(p/2))
            a = 1-b
            q = 1-a*Q11(a,np.sqrt(a**2+(1-a)**2))/(1-a)
            if (x1[i] == (-1)):
                if (x0[i] == (-1)):
                    qb *= q
                    if (u0[i] == 0): 
                        qb *= ((1-p)/(1-p/2)) 
                    else: 
                        qb *= ((p/2)/(1-p/2))
                else:
                    qb *= (1-q)
            else:
                if (x0[i] == (-1)):
                    qb *= (b*(1-q)/(1-b))
                    if (u0[i] == 0):
                        qb *= ((1-p)/(1-p/2))
                    else:
                        qb *= ((p/2)/(1-p/2))
                else:
                    qb *= (1-(b*(1-q)/(1-b)))
    for i in range(2):
        if (u0[i] == 0):
            qf *= (1-p)
        else:
            qf *= (p/2)
        if (u1[i] == 0):
            qb *= (1-p)
        else:
            qb *= (p/2)
    reject_ratio = np.exp(-beta*(np.matrix(xstar)*interacts*np.matrix(xstar).T-np.matrix(x)*interacts*np.matrix(x).T))*qb/qf
    if (np.random.rand()<=reject_ratio):
        x = xstar 
        u = ustar
        acc = 1
    else:
        acc = 0 
    x = -0.5*x+0.5
    x = x.astype('int')
    n = int(''.join(np.array(x).astype('str')), 2)
    return n, u, acc


# if __name__ == '__main__':
#     method  = input('Method for sampling:')
#     nsample = int(input('Total samples for MCMC'))
#     sub_size = int(input('size of sub sample for trace plot'))
#     beta = np.log(float(input('potential parameter')))/4
#     theo_prob = theoretical_val(beta)
#     if (method == 'G'):
#         ns = np.zeros(nsample)
#         n = 3 
#         trace = []
#         x1sum = 0
#         for i in range(40000): 
#             n = three_dim_Gibbs(beta, n) 
#         for i in range(nsample): 
#             n = three_dim_Gibbs(beta, n) 
#             ns[i] = n 
#             if (n%2 == 1): 
#                 x1sum += 1 
#             else: 
#                 x1sum -= 1 
#             if (i%(sub_size) == (sub_size-1)): 
#                 trace.append(x1sum)  
#                 x1sum = 0
#         trace = np.array(trace)
#     else:
#         q = float(input('Momentum Probability'))
#         ns = np.zeros(nsample)
#         u = aux_initialize(q,3)
#         us = np.zeros(nsample)
#         x1sum = 0
#         acc_num = 0
#         n = 3
#         trace = []
#         for i in range(40000):
#             n, u, acc = three_dim_sampling_3(u, beta, q, n)

#         for i in range(nsample):
    
#             n, u, acc = three_dim_sampling_3(u, beta, q, n)
#             acc_num += acc
#             ns[i] = n
#             us[i] = u[0]
#             if (n%2 == 1):
#                 x1sum += 1
#             else:
#                 x1sum -= 1
#             if (i%(sub_size) == (sub_size-1)):
#                 trace.append(x1sum)
#                 x1sum = 0
    
#         trace = np.array(trace)
#         print(acc_num/nsample)
#     plt.plot(np.arange(200), trace[0:200]/sub_size)
#     print(theo_prob)
#     print(np.array(list(collections.Counter(ns).values()))/nsample)
#     print(np.array(list(collections.Counter(us).values()))/nsample)

    
if __name__ == '__main__':
    beta = float(input('please input beta'))
    k = float(input('please input k'))
    nsample = int(input('please input number of sample'))
    p = float(input('please input momentum probability'))
    ns = np.zeros(nsample)
    us = np.zeros(nsample)
    u = aux_initialize(p, 2)
    n = 3 
    acc_num = 0
    for i in range(40000): 
            n, u, acc = two_dim_sampling_1(u, beta, k, p, n)
    for i in range(nsample): 
            n, u, acc = two_dim_sampling_1(u, beta, k, p, n)
            ns[i] = n 
            us[i] = u[0]
            acc_num += acc
    theo_prob = np.array([np.exp(-beta-k), np.exp(beta-k), np.exp(beta+k), np.exp(-beta+k)])
    print(theo_prob/np.sum(theo_prob)) 
    print(np.array(list(collections.Counter(ns).values()))/nsample)
    print(np.array(list(collections.Counter(us).values()))/nsample)

    print(acc_num/nsample)
    
    
 