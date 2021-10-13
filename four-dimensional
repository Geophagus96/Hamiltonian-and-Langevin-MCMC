import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
from copy import deepcopy
from statsmodels.graphics.tsaplots import acf
from scipy import trapz
import scipy.stats as st
from math import sqrt
from scipy.integrate import quad 
import time



#----------------------------------------------------------------------
##  BLOCK OF FUNCTIONS USED IN THE MAIN CODE
#----------------------------------------------------------------------
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
    '''Energy of a given configuration'''
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
        

def inv_operand(x, y):
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
A = np.array([[0,1,1,0],[1,0,0,1],[1,0,0,1],[0,1,1,0]])
xs = np.array([[0,0,0,0],[0,0,0,1],[0,0,1,0],[0,0,1,1],[0,1,0,0],[0,1,0,1],[0,1,1,0],[0,1,1,1],[1,0,0,0],[1,0,0,1],[1,0,1,0],[1,0,1,1],[1,1,0,0],[1,1,0,1],[1,1,1,0],[1,1,1,1]])
xs[xs==0] = -1
quad_ene = np.diag(np.dot(xs,A.dot(xs.T)))/2

#Four-dimensional over-relaxation with block neighbourhood with the momentum grid and without product approximation of probability
def four_dim_noprod_mom(config, u, beta, alpha):
    a = np.random.randint(0,N)
    b = np.random.randint(0,N)
    x1 = config[a,b]
    x2 = config[a,(b+1)%N]
    x3 = config[(a+1)%N,b]
    x4 = config[(a+1)%N,(b+1)%N]
    nb1 = config[(a-1)%N,b] + config[a, (b-1)%N]
    nb2 = config[(a-1)%N,(b+1)%N] + config[a,(b+2)%N]
    nb3 = config[(a+1)%N,(b-1)%N] + config[(a+2)%N,b]
    nb4 = config[(a+1)%N,(b+2)%N] + config[(a+2)%N,(b+1)%N]
    nbs = np.array([nb1,nb2,nb3,nb4])
    u0 = -np.array([u[a,b],u[a,(b+1)%N],u[(a+1)%N,b],u[(a+1)%N,(b+1)%N]])
    x = np.array([x1,x2,x3,x4])
    xbin = 0.5*x+0.5
    y = xbin+u0
    y = y-(y>1)+(y<0)
    y = 2*y-1     
    y = y.astype(int)
    ys =  np.repeat(np.array([y]), 16, axis=0)
    logpxy = np.sum(np.log((1-p/2)*[xs==ys][0]+(p/2)*[xs!=ys][0]),axis=1)
    ps = np.exp(-beta*(np.dot(xs,nbs)+quad_ene)+logpxy)
    st1 = [(ps[0]+ps[1]),(ps[2]+ps[3]),(ps[4]+ps[5]),(ps[6]+ps[7]),(ps[8]+ps[9]),(ps[10]+ps[11]),(ps[12]+ps[13]),(ps[14]+ps[15])]
    st2 = [(st1[0]+st1[1]),(st1[2]+st1[3]),(st1[4]+st1[5]),(st1[6]+st1[7])]
    st3 = [(st2[0]+st2[1]),(st2[2]+st2[3])]
    p0111 = ps[7]/st1[3]
    p0011 = ps[3]/st1[1]
    p1111 = ps[15]/st1[7]
    p1011 = ps[11]/st1[5]
    p1101 = ps[13]/st1[6]
    p0101 = ps[5]/st1[2]
    p0001 = ps[1]/st1[0]
    p1001 = ps[9]/st1[4]
    p011 = st1[3]/st2[1]
    p001 = st1[1]/st2[0]
    p111 = st1[7]/st2[3]
    p101 = st1[5]/st2[2]
    p01 = st2[1]/st3[0]
    p11 = st2[3]/st3[1]
    p1 = st3[1]/(st3[0]+st3[1])
    if (x1==1):
        w1 = p1*np.random.rand()
        if (x2==1):
            w2 = p11*np.random.rand()
            if (x3==1):
                w3 = p111*np.random.rand()
                if (x4==1):
                    w4 = p1111*np.random.rand()
                else:
                    w4 = p1111+(1-p1111)*np.random.rand()
            else:
                w3 = p111+(1-p111)*np.random.rand()
                if (x4==1):
                    w4 = p1101*np.random.rand()
                else:
                    w4 = p1101+(1-p1101)*np.random.rand()
        else:
            w2 = p11+(1-p11)*np.random.rand()
            if (x3==1):
                w3 = p101*np.random.rand()
                if (x4==1):
                    w4 = p1011*np.random.rand()
                else:
                    w4 = p1011+(1-p1011)*np.random.rand()
            else:
                w3 = p101 + (1-p101)*np.random.rand()
                if (x4==1):
                    w4 = p1001*np.random.rand()
                else:
                    w4 = p1001+(1-p1001)*np.random.rand()
    else:
        w1 = p1+(1-p1)*np.random.rand()
        if (x2==1):
            w2 = p11*np.random.rand()
            if (x3==1):
                w3 = p111*np.random.rand()
                if (x4==1):
                    w4 = p1111*np.random.rand()
                else:
                    w4 = p1111+(1-p1111)*np.random.rand()
            else:
                w3 = p111+(1-p111)*np.random.rand()
                if (x4==1):
                    w4 = p1101*np.random.rand()
                else:
                    w4 = p1101+(1-p1101)*np.random.rand()
        else:
            w2 = p11+(1-p11)*np.random.rand()
            if (x3==1):
                w3 = p001*np.random.rand()
                if (x4==1):
                    w4 = p0011*np.random.rand()
                else:
                    w4 = p0011+(1-p1011)*np.random.rand()
            else:
                w3 = p001 + (1-p001)*np.random.rand()
                if (x4==1):
                    w4 = p0001*np.random.rand()
                else:
                    w4 = p0001+(1-p0001)*np.random.rand()
    u1new = (w1+alpha*np.random.rand())%1
    u2new = (w2+alpha*np.random.rand())%1
    u3new = (w3+alpha*np.random.rand())%1
    u4new = (w4+alpha*np.random.rand())%1
    if (u1new<=p1):
        x1new = 1
        if (u2new<=p11):
            x2new = 1
            if (u3new<=p111):
                x3new = 1
                if (u4new<=p1111):
                    x4new = 1
                else:
                    x4new = (-1)
            else:
                x3new = (-1)
                if (u4new<=p1101):
                    x4new = 1
                else:
                    x4new = (-1)
        else:
            x2new = (-1)
            if (u3new<=p101):
                x3new = 1
                if (u4new <= p1011):
                    x4new = 1
                else:
                    x4new = (-1)
            else:
                x3new = (-1)
                if (u4new <= p1001):
                    x4new = 1
                else:
                    x4new = (-1)
    else:
        x1new = (-1)
        if (u2new<=p01):
            x2new = 1
            if (u3new<=p011):
                x3new = 1
                if (u4new<=p0111):
                    x4new = 1
                else:
                    x4new = (-1)
            else:
                x3new = (-1)
                if (u4new<=p0101):
                    x4new = 1
                else:
                    x4new = (-1)
        else:
            x2new = (-1)
            if (u3new<=p001):
                x3new = 1
                if (u4new <= p0011):
                    x4new = 1
                else:
                    x4new = (-1)
            else:
                x3new = (-1)
                if (u4new <= p0001):
                    x4new = 1
                else:
                    x4new = (-1)
    config[a,b] = x1new
    config[a,(b+1)%N] = x2new
    config[(a+1)%N,b] = x3new
    config[(a+1)%N,(b+1)%N] = x4new
    unew = [0,0,0,0]
    xnew = [x1new, x2new, x3new, x4new]
    for i in range(4):
        unew[i] = inv_operand(xnew[i],y[i])
    u[a,b] = unew[0]
    u[a,(b+1)%N] = unew[1]
    u[(a+1)%N,b] = unew[2]
    u[(a+1)%N,(b+1)%N] = unew[3]
    return config, u

#Four-dimensional over-relaxation with block neighbourhood without product approximation 
def four_dim_noprod(config, beta, alpha):
    a = np.random.randint(0,N)
    b = np.random.randint(0,N)
    x1 = config[a,b]
    x2 = config[a,(b+1)%N]
    x3 = config[(a+1)%N,b]
    x4 = config[(a+1)%N,(b+1)%N]
    nb1 = config[(a-1)%N,b] + config[a, (b-1)%N]
    nb2 = config[(a-1)%N,(b+1)%N] + config[a,(b+2)%N]
    nb3 = config[(a+1)%N,(b-1)%N] + config[(a+2)%N,b]
    nb4 = config[(a+1)%N,(b+2)%N] + config[(a+2)%N,(b+1)%N]
    nbs = np.array([nb1,nb2,nb3,nb4])
    ps = np.exp(-beta*(np.dot(xs,nbs)+quad_ene))
    st1 = [(ps[0]+ps[1]),(ps[2]+ps[3]),(ps[4]+ps[5]),(ps[6]+ps[7]),(ps[8]+ps[9]),(ps[10]+ps[11]),(ps[12]+ps[13]),(ps[14]+ps[15])]
    st2 = [(st1[0]+st1[1]),(st1[2]+st1[3]),(st1[4]+st1[5]),(st1[6]+st1[7])]
    st3 = [(st2[0]+st2[1]),(st2[2]+st2[3])]
    p0111 = ps[7]/st1[3]
    p0011 = ps[3]/st1[1]
    p1111 = ps[15]/st1[7]
    p1011 = ps[11]/st1[5]
    p1101 = ps[13]/st1[6]
    p0101 = ps[5]/st1[2]
    p0001 = ps[1]/st1[0]
    p1001 = ps[9]/st1[4]
    p011 = st1[3]/st2[1]
    p001 = st1[1]/st2[0]
    p111 = st1[7]/st2[3]
    p101 = st1[5]/st2[2]
    p01 = st2[1]/st3[0]
    p11 = st2[3]/st3[1]
    p1 = st3[1]/(st3[0]+st3[1])
    if (x1==1):
        u1 = p1*np.random.rand()
        if (x2==1):
            u2 = p11*np.random.rand()
            if (x3==1):
                u3 = p111*np.random.rand()
                if (x4==1):
                    u4 = p1111*np.random.rand()
                else:
                    u4 = p1111+(1-p1111)*np.random.rand()
            else:
                u3 = p111+(1-p111)*np.random.rand()
                if (x4==1):
                    u4 = p1101*np.random.rand()
                else:
                    u4 = p1101+(1-p1101)*np.random.rand()
        else:
            u2 = p11+(1-p11)*np.random.rand()
            if (x3==1):
                u3 = p101*np.random.rand()
                if (x4==1):
                    u4 = p1011*np.random.rand()
                else:
                    u4 = p1011+(1-p1011)*np.random.rand()
            else:
                u3 = p101 + (1-p101)*np.random.rand()
                if (x4==1):
                    u4 = p1001*np.random.rand()
                else:
                    u4 = p1001+(1-p1001)*np.random.rand()
    else:
        u1 = p1+(1-p1)*np.random.rand()
        if (x2==1):
            u2 = p11*np.random.rand()
            if (x3==1):
                u3 = p111*np.random.rand()
                if (x4==1):
                    u4 = p1111*np.random.rand()
                else:
                    u4 = p1111+(1-p1111)*np.random.rand()
            else:
                u3 = p111+(1-p111)*np.random.rand()
                if (x4==1):
                    u4 = p1101*np.random.rand()
                else:
                    u4 = p1101+(1-p1101)*np.random.rand()
        else:
            u2 = p11+(1-p11)*np.random.rand()
            if (x3==1):
                u3 = p001*np.random.rand()
                if (x4==1):
                    u4 = p0011*np.random.rand()
                else:
                    u4 = p0011+(1-p1011)*np.random.rand()
            else:
                u3 = p001 + (1-p001)*np.random.rand()
                if (x4==1):
                    u4 = p0001*np.random.rand()
                else:
                    u4 = p0001+(1-p0001)*np.random.rand()
    u1new = (u1+alpha*np.random.rand())%1
    u2new = (u2+alpha*np.random.rand())%1
    u3new = (u3+alpha*np.random.rand())%1
    u4new = (u4+alpha*np.random.rand())%1
    if (u1new<=p1):
        x1new = 1
        if (u2new<=p11):
            x2new = 1
            if (u3new<=p111):
                x3new = 1
                if (u4new<=p1111):
                    x4new = 1
                else:
                    x4new = (-1)
            else:
                x3new = (-1)
                if (u4new<=p1101):
                    x4new = 1
                else:
                    x4new = (-1)
        else:
            x2new = (-1)
            if (u3new<=p101):
                x3new = 1
                if (u4new <= p1011):
                    x4new = 1
                else:
                    x4new = (-1)
            else:
                x3new = (-1)
                if (u4new <= p1001):
                    x4new = 1
                else:
                    x4new = (-1)
    else:
        x1new = (-1)
        if (u2new<=p01):
            x2new = 1
            if (u3new<=p011):
                x3new = 1
                if (u4new<=p0111):
                    x4new = 1
                else:
                    x4new = (-1)
            else:
                x3new = (-1)
                if (u4new<=p0101):
                    x4new = 1
                else:
                    x4new = (-1)
        else:
            x2new = (-1)
            if (u3new<=p001):
                x3new = 1
                if (u4new <= p0011):
                    x4new = 1
                else:
                    x4new = (-1)
            else:
                x3new = (-1)
                if (u4new <= p0001):
                    x4new = 1
                else:
                    x4new = (-1)
    config[a,b] = x1new
    config[a,(b+1)%N] = x2new
    config[(a+1)%N,b] = x3new
    config[(a+1)%N,(b+1)%N] = x4new
    return config
