runfile('theoretical_val.py')
runfile('four-dimensional.py')

#Grid setup
N = 10         #  size of the lattice, N x N
nt      = 32         #  number of temperature points
eqSteps = 2000       #  number of MC sweeps for equilibration
mcSteps = 4000       #  number of MC sweeps for calculation
T       = np.linspace(1.53, 3.28, nt); 
n1, n2  = 1.0/(mcSteps*N*N), 1.0/(mcSteps*mcSteps*N*N) 

MSE2_energy = np.zeros([9,9])
MSE2_heat = np.zeros([9,9])
#Parameter Choice for four-dimensional over-relaxation with momentum grid
alphas = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
paux = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
for k in range(9):
 for j in range(9):
    E,M,C,X = np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt)
    for tt in range(nt):
      E1 = M1 = E2 = M2 = 0
      config = initialstate(N)
      u = aux_initialize(N, paux[k])
      iT=1.0/T[tt]; iT2=iT*iT;
    
      for i in range(eqSteps):         # equilibrate
         four_dim_noprod_mom(config, u, iT, paux[k], alphas[j])        # Monte Carlo moves

      for i in range(mcSteps):
         four_dim_noprod_mom(config, u, iT, paux[k], alphas[j])          
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
    MSE2_energy[j,k] = np.sum(np.power((E-E_theo),2))
    MSE2_heat[j,k] = np.sum(np.power((C-C_theo),2))
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
    
    plt.savefig('fournoprodmom'+str(k)+str(j)+'.png', dpi=300, bbox_inches='tight')


MSE_energy = np.zeros(9)
MSE_heat = np.zeros(9)
#Parameter Choice for four-dimensional with block neighbourhood over-relaxation
for j in range(9):
    for tt in range(nt):
      E1 = M1 = E2 = M2 = 0
      config = initialstate(N)
      iT=1.0/T[tt]; iT2=iT*iT;
    
      for i in range(eqSteps):         # equilibrate
         four_dim_noprod(config, iT, alphas[j])        # Monte Carlo moves

      for i in range(mcSteps):
         four_dim_noprod(config, iT, alphas[j])          
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
    
    MSE_energy[j] = np.sum(np.power((E-E_theo),2))
    MSE_heat[j] = np.sum(np.power((C-C_theo),2))
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
    
    plt.savefig('fournoprod'+str(j)+'.png', dpi=300, bbox_inches='tight')
