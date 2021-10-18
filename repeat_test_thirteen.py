runfile('theoretical_val.py')

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 14:31:41 2021

@author: Yuze Zhou
"""

#Grid_setup
N = 20         #  size of the lattice, N x N
nt      = 32         #  number of temperature points
eqSteps = 5000       #  number of MC sweeps for equilibration
mcSteps = 10000       #  number of MC sweeps for calculation
T       = np.linspace(1.53, 3.28, nt); 
n1, n2  = 1.0/(mcSteps*N*N), 1.0/(mcSteps*mcSteps*N*N) 


#Repeated Test for four-dimensional with block neighbourhood over-relaxation
alpha = 0.6
energies_or_thirteen = np.zeros([100,nt])
heats_or_thirteen = np.zeros([100,nt])
elpases_or_thirteen = np.zeros(100)

for j in range(100):
    t0 = time.time()
    T       = np.linspace(1.53, 3.28, nt); 
    E,M,C,X = np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt)
    n1, n2  = 1.0/(mcSteps*N*N), 1.0/(mcSteps*mcSteps*N*N) 
    for tt in range(nt):
      E1 = M1 = E2 = M2 = 0
      config = initialstate(N)
      iT=1.0/T[tt]; iT2=iT*iT;
      Ene = calcEnergy(config)
      Mag = calcMag(config)
      for i in range(eqSteps):         # equilibrate
         config, Ene, Mag = thirteen_noprod_cond(config, iT, alpha, Ene, Mag )        # Monte Carlo moves
      Ene = calcEnergy(config)
      Mag = calcMag(config)
      E1 = E1 + Ene
      M1 = M1 + Mag
      M2 = M2 + Mag*Mag 
      E2 = E2 + Ene*Ene
      for i in range(mcSteps):
         config ,Ene, Mag = thirteen_noprod_cond(config, iT, alpha, Ene, Mag)          
         E1 = E1 + Ene
         M1 = M1 + Mag
         M2 = M2 + Mag**2
         E2 = E2 + Ene**2

      E[tt] = n1*E1
      M[tt] = n1*M1
      C[tt] = (n1*E2 - n2*E1*E1)*iT2
      X[tt] = (n1*M2 - n2*M1*M1)*iT
    energies_or[j,] = E
    heats_or[j,] = C
    elpases_or[j] = time.time()-t0

np.sum(np.power((np.mean(energies_or, axis=0)-E_theo),2))
np.sum(np.var(energies_or, axis=0))
np.sum(np.mean(np.power(energies_or-E_theo,2),axis=0))

np.sum(np.power((np.mean(heats_or, axis=0)-C_theo),2));
np.sum(np.var(heats_or, axis=0))
np.sum(np.mean(np.power(heats_or-C_theo,2),axis=0))

#Repeated test for four-dimensional with block neighbourhood and momentum grid over_relaxation
p1 = 0.8
alpha = 0.4
energies_mom_13 = np.zeros([100,nt])
heats_mom_13 = np.zeros([100,nt])
elpases_mom_13 = np.zeros(100)

for j in range(100):
    t0 = time.time()
    T       = np.linspace(1.53, 3.28, nt); 
    E,M,C,X = np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt)
    n1, n2  = 1.0/(mcSteps*N*N), 1.0/(mcSteps*mcSteps*N*N) 
    for tt in range(nt):
      E1 = M1 = E2 = M2 = 0
      config = initialstate(N)
      u = aux_initialize(N,p1)
      iT=1.0/T[tt]; iT2=iT*iT;
      Ene = calcEnergy(config)
      Mag = calcMag(config)
      for i in range(eqSteps):         # equilibrate
         config, u, Ene, Mag = thirteen_mom_cond(config, u, iT, alpha, p1, Ene, Mag)        # Monte Carlo moves
      Ene = calcEnergy(config)
      Mag = calcMag(config)
      E1 = E1 + Ene
      M1 = M1 + float(Mag)
      M2 = M2 + float(Mag*Mag) 
      E2 = E2 + Ene*Ene
      for i in range(mcSteps):
         config, u, Ene, Mag = thirteen_mom_cond(config, u, iT, alpha, p1, Ene, Mag)          
         E1 = E1 + Ene
         M1 = M1 + float(Mag)
         M2 = M2 + float(Mag*Mag) 
         E2 = E2 + Ene*Ene

      E[tt] = n1*E1
      M[tt] = n1*M1
      C[tt] = (n1*E2 - n2*E1*E1)*iT2
      X[tt] = (n1*M2 - n2*M1*M1)*iT
    energies_mom_13[j,] = E
    heats_mom_13[j,] = C
    elpases_mom_13[j] = time.time()-t0

np.sum(np.power((np.mean(energies_mom, axis=0)-E_theo),2))
np.sum(np.var(energies_mom, axis=0))
np.sum(np.mean(np.power(energies_mom-E_theo,2),axis=0))

np.sum(np.power((np.mean(heats_mom, axis=0)-C_theo),2))
np.sum(np.var(heats_mom, axis=0))
np.sum(np.mean(np.power(heats_mom-C_theo,2),axis=0))

#Repeated test for four-dimensional with block neighbourhood discrete-MALA
energies_Gibbs = np.zeros([100,nt])
heats_Gibbs = np.zeros([100,nt])
elpases_Gibbs = np.zeros(100)

for j in range(100):
    t0 = time.time()
    T       = np.linspace(1.53, 3.28, nt); 
    E,M,C,X = np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt)
    n1, n2  = 1.0/(mcSteps*N*N), 1.0/(mcSteps*mcSteps*N*N) 
    for tt in range(nt):
      E1 = M1 = E2 = M2 = 0
      config = initialstate(N)
      iT=1.0/T[tt]; iT2=iT*iT;
    
      for i in range(eqSteps):         # equilibrate
         thirteen_Gibbs(config, iT)        # Monte Carlo moves

      for i in range(mcSteps):
         thirteen_Gibbs(config, iT)          
         Ene = calcEnergy(config)     # calculate the energy
         Mag = calcMag(config)        # calculate the magnetisation
         E1 = E1 + Ene
         M1 = M1 + Mag
         M2 = M2 + Mag*Mag 
         E2 = E2 + Ene*Ene

      E[tt] = n1*E1
      M[tt] = n1*M1
      C[tt] = (n1*E2 - n2*E1*E1)*iT2
      X[tt] = (n1*M2 - n2*M1*M1)*iT
    energies_Gibbs[j,] = E
    heats_Gibbs[j,] = C
    elpases_Gibbs[j] = time.time()-t0

np.sum(np.power((np.mean(energies_Gibbs, axis=0)-E_theo),2))
np.sum(np.var(energies_Gibbs, axis=0))
np.sum(np.mean(np.power(energies_Gibbs-E_theo,2),axis=0))

np.sum(np.power((np.mean(heats_Gibbs, axis=0)-C_theo),2))
np.sum(np.var(heats_Gibbs, axis=0))
np.sum(np.mean(np.power(heats_Gibbs-C_theo,2),axis=0))

plt.plot(T, np.var(energies_or, axis=0), label='over-relaxation');plt.plot(T, np.var(energies_mom, axis=0), label='momentum');plt.plot(T, np.var(energies_Gibbs,axis=0),label='Gibbs');plt.legend();plt.show()
plt.plot(T, np.var(heats_or, axis=0), label='over-relaxation');plt.plot(T, np.var(heats_mom, axis=0), label='momentum');plt.plot(T, np.var(heats_Gibbs,axis=0),label='Gibbs');plt.legend();plt.show()


