
runfile('theoretical_val.py')
runfile('four-dimensional.py')

#Grid_setup
N = 10         #  size of the lattice, N x N
nt      = 32         #  number of temperature points
eqSteps = 2000       #  number of MC sweeps for equilibration
mcSteps = 4000       #  number of MC sweeps for calculation
T       = np.linspace(1.53, 3.28, nt); 
n1, n2  = 1.0/(mcSteps*N*N), 1.0/(mcSteps*mcSteps*N*N) 


#Repeated Test for four-dimensional with block neighbourhood over-relaxation
alpha = 0.8
energies_or = np.zeros([100,nt])
heats_or = np.zeros([100,nt])
elpases_or = np.zeros(100)

for j in range(100):
    t0 = time.time()
    Energies_dmala = np.zeros([nt, mcSteps])
    T       = np.linspace(1.53, 3.28, nt); 
    E,M,C,X = np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt)
    n1, n2  = 1.0/(mcSteps*N*N), 1.0/(mcSteps*mcSteps*N*N) 
    for tt in range(nt):
      E1 = M1 = E2 = M2 = 0
      config = initialstate(N)
      iT=1.0/T[tt]; iT2=iT*iT;
    
      for i in range(eqSteps):         # equilibrate
         four_dim_noprod(config, iT, alpha)        # Monte Carlo moves

      for i in range(mcSteps):
         four_dim_noprod(config, iT, alpha)          
         Ene = calcEnergy(config)     # calculate the energy
         Mag = calcMag(config)        # calculate the magnetisation
         Energies_dmala[tt,i] = Ene
         E1 = E1 + Ene
         M1 = M1 + Mag
         M2 = M2 + Mag*Mag 
         E2 = E2 + Ene*Ene

      E[tt] = n1*E1
      M[tt] = n1*M1
      C[tt] = (n1*E2 - n2*E1*E1)*iT2
      X[tt] = (n1*M2 - n2*M1*M1)*iT
    energies_or[j,] = E
    heats_or[j,] = C
    elpases_or[j] = time.time()-t0

#Repeated test for four-dimensional with block neighbourhood and momentum grid over_relaxation
p1 = 0.3
alpha = 0.8
energies_mom = np.zeros([100,nt])
heats_mom = np.zeros([100,nt])
elpases_mom = np.zeros(100)

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
         four_dim_noprod(config, iT, alpha)        # Monte Carlo moves

      for i in range(mcSteps):
         four_dim_noprod(config, iT, alpha)          
         Ene = calcEnergy(config)     # calculate the energy
         Mag = calcMag(config)        # calculate the magnetisation
         Energies_dmala[tt,i] = Ene
         E1 = E1 + Ene
         M1 = M1 + Mag
         M2 = M2 + Mag*Mag 
         E2 = E2 + Ene*Ene

      E[tt] = n1*E1
      M[tt] = n1*M1
      C[tt] = (n1*E2 - n2*E1*E1)*iT2
      X[tt] = (n1*M2 - n2*M1*M1)*iT
    energies_mom[j,] = E
    heats_mom[j,] = C
    elpases_mom[j] = time.time()-t0

#Repeated test for four-dimensional with block neighbourhood discrete-MALA
energies_mala = np.zeros([100,nt])
heats_mala = np.zeros([100,nt])
elpases_mala = np.zeros(100)

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
         dmala(config, iT)        # Monte Carlo moves

      for i in range(mcSteps):
         dmala(config, iT)          
         Ene = calcEnergy(config)     # calculate the energy
         Mag = calcMag(config)        # calculate the magnetisation
         Energies_dmala[tt,i] = Ene
         E1 = E1 + Ene
         M1 = M1 + Mag
         M2 = M2 + Mag*Mag 
         E2 = E2 + Ene*Ene

      E[tt] = n1*E1
      M[tt] = n1*M1
      C[tt] = (n1*E2 - n2*E1*E1)*iT2
      X[tt] = (n1*M2 - n2*M1*M1)*iT
    energies_mom[j,] = E
    heats_mom[j,] = C
    elpases_mom[j] = time.time()-t0
