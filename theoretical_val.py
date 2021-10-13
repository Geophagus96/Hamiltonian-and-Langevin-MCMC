# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 23:18:46 2021

@author: Yuze Zhou
"""

import numpy as np
from scipy.misc import derivative
from scipy.integrate import quad 
from math import sinh, cosh, log, sin, pi
import matplotlib.pyplot as plt

def lnlambda(beta):
    K = (2*sinh(-2*beta))/((cosh(-2*beta))**2)
    def integrand(w):
        return log(0.5*(1+(1-(K*sin(w))**2)**(0.5)))
    return log(2*cosh(-2*beta)) + (1/pi)*quad(integrand, 0, pi/2)[0]

def derivelambda(beta):
     return derivative(lnlambda, beta, dx=1e-10)
 

nt = 32
T = np.linspace(1.53, 3.28, nt)
E_theo = np.zeros(nt)
C_theo = np.zeros(nt)
for i in range(32):
    E_theo[i] = derivative(lnlambda, 1.0/T[i], dx=1e-8)/2
    C_theo[i] =  (1.0/T[i])**2*derivative(derivelambda, 1.0/T[i], dx=1e-5)/4


plt.plot(T, C_theo, 'o')