# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 15:53:21 2022

@author: Yuze Zhou
"""

#Grid_setup
N = 20         #  size of the lattice, N x N
nt      = 32         #  number of temperature points
T       = np.linspace(1.53, 3.28, nt); 
#trace plots of energies for single flips 
trace_num = 15000
trace_single_mom = np.zeros([nt, trace_num])
p = 0.3
for tt in range(nt):
    iT=1.0/T[tt]
    config_mom = initialstate(N)
    u_mom = aux_initialize(N, p)
    Ene = calcEnergy(config_mom)
    Mag = calcMag(config_mom)
    for j in range(trace_num):
        config_mom, u_mom, Ene, Mag = thirteen_mom_optimal_alpha(config_mom, u_mom, iT, p, Ene, Mag)
        trace_single_mom[tt, j] = calcEnergy_new(config_mom)

    
mean_E = 400*E_theo
        
#Neighourhood size comparison (1, 15, 29)
plt.plot(np.arange(trace_num), trace_single_mom[1,], label='momemtum_single');plt.plot(np.arange(trace_num), np.repeat(mean_E[1], trace_num), color = 'red');plt.legend();plt.show()
