# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 17:33:27 2021

@author: Yuze Zhou
"""

import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
from copy import deepcopy

#----------------------------------------------------------------------
##  BLOCK OF FUNCTIONS USED IN THE MAIN CODE
#----------------------------------------------------------------------
def initialstate(N):   
    ''' generates a random spin configuration for initial condition'''
    state = 2*np.random.randint(2, size=(N,N))-1
    return state

def calcEnergy(config):
    '''Energy of a given configuration'''
    energy = 0
    for i in range(len(config)):
        for j in range(len(config)):
            S = config[i,j]
            nb = config[(i+1)%N, j] + config[i,(j+1)%N] + config[(i-1)%N, j] + config[i,(j-1)%N]
            energy += -nb*S
    return energy/4.

def random_gibbs(config, beta):
    for k in range(N):
        for l in range(N):
            a = np.random.randint(0,N)
            b = np.random.randint(0,N)
            s = config[a,b]
            energy = np.exp(-2*beta*calcEnergy(config))
            new_config = deepcopy(config)
            new_config[a,b] = s*(-1)
            new_energy = np.exp(-2*beta*calcEnergy(new_config))
            p = new_energy/(energy+new_energy)
            u = rand()
            if (u<=p):
                config[a,b] = s*(-1)
    return config

def random_gibbs_single(config, beta):
     a = np.random.randint(0,N)
     b = np.random.randint(0,N)
     s = config[a,b]
     energy = np.exp(-2*beta*calcEnergy(config))
     new_config = deepcopy(config)
     new_config[a,b] = s*(-1)
     new_energy = np.exp(-2*beta*calcEnergy(new_config))
     p = new_energy/(energy+new_energy)
     u = rand()
     if (u<=p):
         config[a,b] = s*(-1)
     return config

def local_dmala(config, beta):
    for k in range(N):
        for l in range(N):
            a = np.random.randint(0,N)
            b = np.random.randint(0,N)
            u1 = rand()  
            k1 = int(4*u1)
            s1 = config[(a+1)%N,b]
            s2 = config[(a-1)%N,b]
            s3 = config[a,(b+1)%N]
            s4 = config[a,(b-1)%N]
            if (k1==0):
                config[(a+1)%N,b] = s1*(-1)
            elif (k1==1):
                config[(a-1)%N,b] = s2*(-1) 
            elif (k1==2): 
                config[a,(b+1)%N] = s3*(-1) 
            else: 
                config[a,(b-1)%N] = s4*(-1) 
            new_config_1 = deepcopy(config) 
            new_config_2 = deepcopy(config) 
            new_config_3 = deepcopy(config) 
            new_config_4 = deepcopy(config) 
            s1 = new_config_1[(a+1)%N,b] 
            s2 = new_config_2[(a-1)%N,b] 
            s3 = new_config_3[a,(b+1)%N] 
            s4 = new_config_4[a,(b-1)%N] 
            new_config_1[(a+1)%N,b] = s1*(-1) 
            new_config_2[(a-1)%N,b] = s2*(-1) 
            new_config_3[a,(b+1)%N] = s3*(-1) 
            new_config_4[a,(b-1)%N] = s4*(-1) 
            p1 = np.exp(-2*beta*calcEnergy(new_config_1)) 
            p2 = np.exp(-2*beta*calcEnergy(new_config_2)) 
            p3 = np.exp(-2*beta*calcEnergy(new_config_3)) 
            p4 = np.exp(-2*beta*calcEnergy(new_config_4)) 
            subtotal = [p1,p2,p3,p4] 
            subtotal = np.array(subtotal)  
            total = np.sum(subtotal)  
            u = rand()   
            proportion = 0   
            k = -1   
            while proportion <= u:  
                k += 1 
                proportion += subtotal[k]/total 
            if (k==0): 
                config[(a+1)%N,b] = s1*(-1) 
            elif (k==1): 
                config[(a-1)%N,b] = s2*(-1)
            elif (k==2):
                config[a,(b+1)%N] = s3*(-1) 
            else:
                config[a,(b-1)%N] = s4*(-1)    
    return config

def local_dmala_single(config, beta):
    a = np.random.randint(0,N)
    b = np.random.randint(0,N)
    u1 = rand()  
    k1 = int(4*u1)
    s1 = config[(a+1)%N,b]
    s2 = config[(a-1)%N,b]
    s3 = config[a,(b+1)%N]
    s4 = config[a,(b-1)%N]
    if (k1==0):
        config[(a+1)%N,b] = s1*(-1)
    elif (k1==1):
        config[(a-1)%N,b] = s2*(-1)
    elif (k1==2):
        config[a,(b+1)%N] = s3*(-1)
    else:
        config[a,(b-1)%N] = s4*(-1)
    new_config_1 = deepcopy(config)
    new_config_2 = deepcopy(config)
    new_config_3 = deepcopy(config)
    new_config_4 = deepcopy(config)
    s1 = new_config_1[(a+1)%N,b]
    s2 = new_config_2[(a-1)%N,b]
    s3 = new_config_3[a,(b+1)%N]
    s4 = new_config_4[a,(b-1)%N]
    new_config_1[(a+1)%N,b] = s1*(-1)
    new_config_2[(a-1)%N,b] = s2*(-1)
    new_config_3[a,(b+1)%N] = s3*(-1)
    new_config_4[a,(b-1)%N] = s4*(-1)
    p1 = np.exp(-2*beta*calcEnergy(new_config_1))
    p2 = np.exp(-2*beta*calcEnergy(new_config_2))
    p3 = np.exp(-2*beta*calcEnergy(new_config_3))
    p4 = np.exp(-2*beta*calcEnergy(new_config_4))
    subtotal = [p1,p2,p3,p4]
    subtotal = np.array(subtotal) 
    total = np.sum(subtotal) 
    u = rand()  
    proportion = 0  
    k = -1  
    while proportion <= u:  
        k += 1 
        proportion += subtotal[k]/total 
    if (k==0):
        config[(a+1)%N,b] = s1*(-1)
    elif (k==1):
        config[(a-1)%N,b] = s2*(-1)
    elif (k==2):
        config[a,(b+1)%N] = s3*(-1)
    else:
        config[a,(b-1)%N] = s4*(-1)    
    return config

def dmala(config, beta):
    for k in range(N):
        for l in range(N): 
            a = np.random.randint(0,N) 
            b = np.random.randint(0,N) 
            s = config[a,b] 
            config[a,b] = s*(-1) 
            subtotal = [] 
            for i in range(N): 
                for j in range(N): 
                    new_config = deepcopy(config) 
                    s = new_config[i,j] 
                    new_config[i,j] = s*(-1) 
                    subtotal.append(np.exp(-2*beta*calcEnergy(new_config))) 
            subtotal = np.array(subtotal) 
            total = np.sum(subtotal) 
            u = rand()  
            proportion = 0  
            k = -1  
            while proportion <= u:  
                k += 1 
                proportion += subtotal[k]/total 
            i = int(k/N) 
            j = k%N 
            s = config[i,j] 
            config[i,j] = s*(-1)
    return config

def dmala_single(config, beta):
     a = np.random.randint(0,N) 
     b = np.random.randint(0,N) 
     s = config[a,b] 
     config[a,b] = s*(-1) 
     subtotal = [] 
     for i in range(N): 
         for j in range(N): 
             new_config = deepcopy(config) 
             s = new_config[i,j] 
             new_config[i,j] = s*(-1) 
             subtotal.append(np.exp(-2*beta*calcEnergy(new_config))) 
     subtotal = np.array(subtotal) 
     total = np.sum(subtotal) 
     u = rand()  
     proportion = 0  
     k = -1  
     while proportion <= u:  
         k += 1 
         proportion += subtotal[k]/total 
     i = int(k/N) 
     j = k%N 
     s = config[i,j] 
     config[i,j] = s*(-1)
     return config
    
def mcmove(config, beta):
    '''Monte Carlo move using Metropolis algorithm '''
    for i in range(N):
        for j in range(N):
            a = np.random.randint(0, N)
            b = np.random.randint(0, N)
            s =  config[a, b]
            nb = config[(a+1)%N,b] + config[a,(b+1)%N] + config[(a-1)%N,b] + config[a,(b-1)%N] 
            cost = 2*s*nb 
            if cost < 0: 
                s *= -1
            elif rand() < np.exp(-cost*beta): 
                s *= -1
            config[a, b] = s
    return config

def mcmove_single(config, beta):
    a = np.random.randint(0, N)
    b = np.random.randint(0, N)
    s =  config[a, b]
    nb = config[(a+1)%N,b] + config[a,(b+1)%N] + config[(a-1)%N,b] + config[a,(b-1)%N] 
    cost = 2*s*nb 
    if cost < 0: 
        s *= -1
    elif rand() < np.exp(-cost*beta): 
        s *= -1
    config[a, b] = s
    return config
    
def calcMag(config):
    '''Magnetization of a given configuration'''
    mag = np.sum(config)
    return mag

nt      = 32         #  number of temperature points
N       = 10         #  size of the lattice, N x N
eqSteps = 40       #  number of MC sweeps for equilibration
mcSteps = 40000       #  number of MC sweeps for calculation

T       = np.linspace(1.53, 3.28, nt); 
E,M,C,X = np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt)
n1, n2  = 1.0/(mcSteps*N*N), 1.0/(mcSteps*mcSteps*N*N) 
# divide by number of samples, and by system size to get intensive values
#----------------------------------------------------------------------
#  MAIN PART OF THE CODE
#----------------------------------------------------------------------
margins_0 = 2
margins_1 = 3
margin_ts = np.zeros([nt,mcSteps])
for tt in range(nt):
    E1 = M1 = E2 = M2 = 0
    config = initialstate(N)
    iT=1.0/T[tt]; iT2=iT*iT;
    
    for i in range(eqSteps):         # equilibrate
        local_dmala(config, iT)           # Monte Carlo moves

    for i in range(mcSteps):
        local_dmala_single(config, iT)           
        Ene = calcEnergy(config)     # calculate the energy
        Mag = calcMag(config)        # calculate the magnetisation

        E1 = E1 + Ene
        M1 = M1 + Mag
        M2 = M2 + Mag*Mag 
        E2 = E2 + Ene*Ene
        margin_ts[tt,i] = config[margins_0,margins_1]

    E[tt] = n1*E1
    M[tt] = n1*M1
    C[tt] = (n1*E2 - n2*E1*E1)*iT2
    X[tt] = (n1*M2 - n2*M1*M1)*iT

# E_multi,M_multi,C_multi,X_multi = np.zeros([5,nt]), np.zeros([5,nt]), np.zeros([5,nt]), np.zeros([5,nt])
# for k in range(5):
#    for tt in range(nt):
#     E1 = M1 = E2 = M2 = 0
#     config = initialstate(N)
#     iT=1.0/T[tt]; iT2=iT*iT;
    
#     for i in range(eqSteps):         # equilibrate
#         local_dmala(config, iT)           # Monte Carlo moves

#     for i in range(mcSteps):
#         local_dmala_single(config, iT)           
#         Ene = calcEnergy(config)     # calculate the energy
#         Mag = calcMag(config)        # calculate the magnetisation

#         E1 = E1 + Ene
#         M1 = M1 + Mag
#         M2 = M2 + Mag*Mag 
#         E2 = E2 + Ene*Ene
#     E_multi[k,tt] = n1*E1
#     M_multi[k,tt] = n1*M1
#     C_multi[k,tt] = (n1*E2 - n2*E1*E1)*iT2
#     X_multi[k,tt] = (n1*M2 - n2*M1*M1)*iT

# E = np.array(np.mean(E_multi,axis=0))
# M= np.array(np.mean(M_multi,axis=0))
# C = np.array(np.mean(C_multi,axis=0))
# X = np.array(np.mean(X_multi,axis=0))

f = plt.figure(figsize=(18, 10)); # plot the calculated values   
sp =  f.add_subplot(2, 2, 1 );
plt.scatter(T, E, s=50, marker='o', color='IndianRed')
plt.xlabel("Temperature (T)", fontsize=20);
plt.ylabel("Energy ", fontsize=20);         plt.axis('tight');

sp =  f.add_subplot(2, 2, 2 );
plt.scatter(T, abs(M), s=50, marker='o', color='RoyalBlue')
plt.xlabel("Temperature (T)", fontsize=20); 
plt.ylabel("Magnetization ", fontsize=20);   plt.axis('tight');

sp =  f.add_subplot(2, 2, 3 );
plt.scatter(T, C, s=50, marker='o', color='IndianRed')
plt.xlabel("Temperature (T)", fontsize=20);  
plt.ylabel("Specific Heat ", fontsize=20);   plt.axis('tight');   

sp =  f.add_subplot(2, 2, 4 );
plt.scatter(T, X, s=50, marker='o', color='RoyalBlue')
plt.xlabel("Temperature (T)", fontsize=20); 
plt.ylabel("Susceptibility", fontsize=20);   plt.axis('tight');
