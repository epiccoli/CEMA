import sys


from CEMA import *
import pyjacob

import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
import time as t
import pdb

import numpy.linalg as LA
## Plot toggle
plot_EI = True
plot_EI = False
plot_eigenvalue = False
# mode = 'selectedEig' 
mode = 'n_eig'

EI_mode = 'debug'
# EI_mode = 'no debug'




#def find_ind(vect, low_bound, high_bound):

#    idx = np.zeros_like(vect)


#    for i in range(len(vect)):

#        if vect[i] > low_bound and vect[i] < high_bound:
#            idx[i] = 0

#        else:
#            idx[i] = 1

#    return idx



# Initial gas properties
phi = 1.0                   # equivalence ratio
P = 101325                  # pressure [Pa]
T = 1100                    # temperature [K]
fuel_spec = 'H2'            # fuel species

# 0-D reactor simulation parameters
npoints = 1000              # number of timesteps
timestep = 1.5e-7           # timestep 
CEMA_interval = 1 # only divisors of npoints

N_eig = 8                   # number of eigenvalues to store at each timestep
N_EI = 1                    # number of explosive index species 
#### options 
first_eigenmode = True
first_ei = True



# Create gas object
gas = ct.Solution('Li_2003.cti')


## REORDER CANTERA SPECIES AS PYJAC WOULD:
# 1) Open mechanism file and find out which among N2, Argon, Helium is the first species to appear
# 2) Set that as the last species in last_spec variable
last_spec = 'AR'
specs = gas.species()[:]
N2_ind = gas.species_index(last_spec)
gas = ct.Solution(thermo='IdealGas', kinetics='GasKinetics',
        species=specs[:N2_ind] + specs[N2_ind + 1:] + [specs[N2_ind]],
        reactions=gas.reactions())
# check if rearranging of species as pyjac would has been effective
# print gas.species_name(28) # >> should give last_spec


# create list containing the ordered entries in the thermo-chemical state vector 
EI_keys = ['']*gas.n_species

EI_keys[0] = 'T'
for i in range(1,gas.n_species):
    EI_keys[i] = gas.species_name(i-1)
    print gas.species_name(i)
print EI_keys   


## SET INITIAL CONDITIONS
gas.set_equivalence_ratio(phi,fuel_spec,'O2:1, N2:3.76')
gas.TP = T, P

# # Create constant pressure reactor
r = ct.IdealGasConstPressureReactor(gas)

# # Create simulation PSR object
sim = ct.ReactorNet([r])

# # Initialize time and data vectors
time = 0.0

tim = np.zeros(npoints,'d')
temp = np.zeros(npoints,'d')
press = np.zeros(npoints,'d')
enth = np.zeros(npoints,'d')




eigenvalues = np.zeros((N_eig,npoints))
expl_indices = np.zeros((gas.n_species,npoints))
CEM = np.zeros(npoints)



track_species = []

count=0

most_aligned_eig = []
most_aligned_ei = []
alignment = np.zeros(gas.n_species)



start = t.time()


for n in range(npoints):
    time += timestep
    sim.advance(time)
    
    tim[n] = time
    temp[n]= r.T

    D, L, R = solve_eig_gas(gas)    

    eigenvalues[:,n] = D[np.argsort(D)[-N_eig:]]        # store the N_eighighest eigenvalues by real magnitude for each timestep n 


   
    # WHEN LOOKING AT WHICH EI IS WHICH (DEBUG EI FOR PLOTS)
    order=-1
    max_idx = np.argsort(D)[order]
    

    expl_indices[:,n] = EI(D,L,R,max_idx)
    main_EI = np.argsort(expl_indices[:,n])[-N_EI:]

    


end = t.time()



print end-start, ' seconds'


dT=np.diff(temp)
dTdt = dT/np.diff(tim)

selected_species = []

# Time series plot of temperature, maximum eigenvalue, temperature gradient
# plt.figure(figsize=(10,5))
# plt.subplot(3,1,1)
# plt.plot(tim,temp)
# plt.title('Temp')

# plt.subplot(3,1,2)
# plt.plot(tt,np.array(val)/1e6)
# plt.title('Eig')

# plt.subplot(3,1,3)
# plt.plot(np.arange(0,len(dT)),dT)
# plt.title('Temperature gradient')

# plt.show()

# plt.figure()
# plt.plot(tt*1e6,CEM/1e6)
# plt.xlim(0,100)
# plt.ylim(-0.5,0.5)
# plt.show()    




#### EI plot ####

# if plot_EI == True:
#   fig, ax = plt.subplots() 
#   for i in range(len(idx_entries)):
#       plt.plot(tim,expl_indices[idx_entries[i],:],linestyle='--', marker='o',label=track_entries[i])
#   plt.axvline(x=switch_time, ymin=0., ymax = 1, linewidth=1, color='k')
#   plt.xlim((7.5e-5,9e-5))
#   plt.xlabel('Residence time')
#   plt.xticks([7.5e-5, 8e-5, switch_time, 8.5e-5, 9e-5])

#   plt.legend()
#   plt.show()

# if EI_mode == 'debug':
#   track_species=map(int,track_species)
#   fig, ax = plt.subplots(2,1) 
#   plt.subplot(2,1,1)
#   for i in range(len(track_species)):
        
#       plt.plot(tim*1e6,expl_indices[track_species[i],:],linestyle='--', marker='o',label=EI_keys[track_species[i]])
#   # plt.xlim((7.5e-5,9e-5))
#   plt.xlabel('Residence time')
#   plt.axvline(x=switch_time*1e6, ymin=0., ymax = 1, linewidth=1, color='k')
#   # plt.xticks([7.5e-5, 8e-5, switch_time, 8.5e-5, 9e-5])

#   plt.legend()
#   # plt.show()

#   plt.subplot(2,1,2)
#   plt.plot(tim*1e6,eigenvalues[order,:]/1e6,linestyle='--', marker='.')
#   plt.axvline(x=switch_time*1e6, ymin=0., ymax = 1, linewidth=1, color='k')

#   titlefig = str(order) + '_provaEI' + '.pdf'
#   plt.savefig(titlefig, bbox_inches='tight')



#### EIG PLOT ####

if mode == 'n_eig':
    legend_entry = ['8th','7th','6th','5th','4th','3rd','2nd','1st']
    plt.figure()
    for i in range(N_eig):

        # plt.subplot(N_eig,1,i+1) # comment to overlay instead of subplot
        plt.plot(tim*1e6,eigenvalues[i,:]/1e6,linestyle='--',marker='.',label=legend_entry[i])
    plt.plot(tim*1e6,CEM/1e6,'x',label='CEM')
    plt.legend() 
    
    plt.show()



eigenvalues = np.flipud(eigenvalues)

df = pd.DataFrame(
    {'time': tim,
    'eig1': eigenvalues[0,:],
    'eig2': eigenvalues[1,:],
    'eig3': eigenvalues[2,:],
    'eig3': eigenvalues[3,:],
    'eig4': eigenvalues[4,:],
    'eig5': eigenvalues[5,:],
    'eig6': eigenvalues[6,:],
    'eig7': eigenvalues[7,:]
    })
        
# df.to_csv('eigenvalues_autoignition_Li2003_H2.csv')


