import pdb
import sys
sys.path.append('./Skeletal29_N/')       # path to pyjacob.so
import pyjacob
from functions import *

import os
import cantera as ct
import numpy as np
import timeit
import matplotlib.pyplot as plt
import scipy.linalg as LA 
import pandas as pd 

'''
For ROBERTO:

there are a number of things which are not very clear in this script. I am sorry, I could not find a better one between yesterday and today. 
First, most of the functions are in the functions.py file and not in this main script file. For the way how CEMA is implemented in particular, you have to take a look at those functions.

This script first creates and solves a 1D flame cantera object. Then the flame solution is loaded and the eigenvalues are calculated. 
You find a for loop (for eig2track in range(-1, -2, -1)) which is not intuitive: as the script is right now, eig2track only takes the value -1: this corresponds to looking for the maximum eigenvalue and trying to follow its evolution (the Chemical Explosive Mode). 
I had this formulation because I wanted to see what happens if I try to track mode 2, mode 3, etc as well. 

Based on the value of Z (mixture fraction), two different functions are called to do CEMA: one has the appendix "no_update" and the other "update": this corresponds to different strategies that I was trying out to recover the Chemical Explosive Mode.
the update/no-update refers whether the explosive index vector which is compared to the EI at each location is updated progressively or not. In the former case, each grid point is compared to the contiguous one;
in the latter case, every point's EI vector is compared to the EI vector where the maxiumum eigenvalue was found.
'''

#create gas from original mechanism file gri30.cti
gas = ct.Solution('Skeletal29_N.cti')
fuel_species = 'C2H4'


# Eigenvalue to follow
# eig2track = - # 0 for maximum eig
fitting = 'cos'


####### EITHER set the gas state with mix fraction ########
phi_j = 1.2
# Z = 1.0; width = 0.03
Z = 0.07; width = 0.03
gas, phi, T = set_mixture_wagner(gas,Z,phi_j)

flame_filename = 'saved_T={:.0f}_phi={:.4f}_Z={:.3f}_phiJ={:.1f}_W={:.2f}mm.xml'.format(T, phi, Z, phi_j,width*1000)


####### OR set the gas state with stoechiometry #######

# P = 101325 # Pa
# phi = 1.0 # Most reactive mixture fraction
# T = 1400 # K
# flame filename to store it and load it again
# flame_filename = 'saved_T={:.0f}_phi={:.4f}.xml'.format(T,phi)
# gas.set_equivalence_ratio(phi, fuel_species, 'O2:1.0, N2:3.76')
# gas.TP = T,P


# 1D flame simulation
initial_grid = np.linspace(0, width, 7) # coarse initial grid 

f = ct.FreeFlame(gas, grid=initial_grid)

# CREATE FLAME or RESTORE IT (based on if the flame data for the operating point exists)
###############################################################################################
if os.path.isfile(flame_filename):
    f.restore(filename=flame_filename)
else: 
    ### Create Flame object and save it:
    f.set_refine_criteria(ratio=2, slope=0.02, curve=0.05)
    f.solve(loglevel=1, auto=True)
    f.save(filename=flame_filename) #, name='', description='solution of hot stoech. OLI')

print('\nmixture-averaged flamespeed = {:7f} m/s\n'.format(f.u[0]))
###############################################################################################

# Check temperature profile of 1D flame
plt.figure()
plt.plot(f.grid[:],f.T[:])
plt.title('Temperature profile of 1D flame')
plt.xlabel('x [m]')
plt.ylabel('Temperature [K]')
plt.show()


# Track the eigenvalues in the flame solution object (f):

for eig2track in range(-1,-2,-1):#(-1,-10,-1):
    
    graph_filename = 'Mode{}_Z={:.3f}_phi={:.2f}_T={:.0f}_phiJ={:.4f}.pdf'.format(str(abs(eig2track)), Z, phi, T, phi_j)

    ##### CEMA for the flame #####
    # eig_track, global_expl_indices, track_species, max_eig_loc, eigenvalues, hard_points = solve_eig_flame_track_update(f,gas,fitting)
    if Z < 0.1:
        eig_track, global_expl_indices, track_species, max_eig_loc, eigenvalues, hard_points = solve_eig_flame_track_update(f,gas,fitting)
    if Z > 0.1:
        eig_track, global_expl_indices, track_species, max_eig_loc, eigenvalues, hard_points = solve_eig_track_no_update(f,gas,fitting)
    
    



    # Build vector with names of EI vector [T + species] 
    EI_labels = get_names(gas)

    # Split eigenvalue tracked and x points to display visually the forward and backward propagation of the eigenvalue tracking procedure
    forward_grid = f.grid[max_eig_loc:]     # x points in [x_max_eig, x_end]
    backward_grid = f.grid[:max_eig_loc]    # x points in [0, x_max_eig]
    CEM_fw=eig_track[max_eig_loc:]          # eigenvalues at forward_grid locations 
    CEM_bw=eig_track[:max_eig_loc]          # eigenvalues at backward_grid locations

    fig = plt.subplots(figsize=(16,10))
    ax1=plt.subplot(2,1,1)

    ax1.plot(forward_grid, np.array(CEM_fw),linestyle='--',marker='.',label='fw tracking')
    ax1.plot(backward_grid, np.array(CEM_bw),linestyle='--',marker='.',label='bw tracking')
    plt.legend()
    plt.title('Timescale of mode {}'.format(str(abs(eig2track))))

    # plot hard points
    hard_loc = np.where(hard_points==1) # can be used for different marking (see in functions, check_alignement() function )

    ax1.plot(f.grid[hard_loc],np.zeros_like(hard_loc).T,marker='x',color='k')

    ax2 = plt.subplot(2,1,2,sharex=ax1)

    for i in range(len(track_species)):

        ax2.plot(f.grid,global_expl_indices[track_species[i],:],linestyle='--',marker='.',label=EI_labels[track_species[i]])

    plt.legend()    
    plt.title('Most important EI of mode {}'.format(str(abs(eig2track))))

    plt.suptitle('Eigenvalue number {} tracked from flame front'.format(str(abs(eig2track))))
    plt.savefig(graph_filename)
    plt.show()

    # save data in pandas dataframe 
    df = pd.DataFrame(
    {'x': f.grid,           # x locations
     'cem': eig_track,      # eigenvalue tracked
     'T': f.T               # temperature
    })

    if not os.path.exists('../data_graphs/'):
        os.mkdir('../data_graphs/')
    df.to_csv('../data_graphs/'+os.path.splitext(graph_filename)[0]+'.csv')
    print('Written file with EIG information in ../data_graphs')



    ### EXTRACT EI AT MAX EIG LOCATION
    

##### PLOT ALL EIG

# fig = plt.subplots(figsize=(16,10))
# ax1=plt.subplot(8,1,1)
# ax2=plt.subplot(8,1,2,sharex=ax1)
# ax3=plt.subplot(8,1,3,sharex=ax1)
# ax4=plt.subplot(8,1,4,sharex=ax1)
# ax5=plt.subplot(8,1,5,sharex=ax1)
# ax6=plt.subplot(8,1,6,sharex=ax1)
# ax7=plt.subplot(8,1,7,sharex=ax1)
# ax8=plt.subplot(8,1,8,sharex=ax1)


# ax1.plot(f.grid,eigenvalues[7,:],linestyle='--',marker='.',label='1')
# plt.legend()
# ax2.plot(f.grid,eigenvalues[6,:],linestyle='--',marker='.',label='2')
# plt.legend()
# ax3.plot(f.grid,eigenvalues[5,:],linestyle='--',marker='.',label='3')
# plt.legend()
# ax4.plot(f.grid,eigenvalues[4,:],linestyle='--',marker='.',label='4')
# plt.legend()
# ax5.plot(f.grid,eigenvalues[3,:],linestyle='--',marker='.',label='5')
# plt.legend()
# ax6.plot(f.grid,eigenvalues[2,:],linestyle='--',marker='.',label='6')
# plt.legend()
# ax7.plot(f.grid,eigenvalues[1,:],linestyle='--',marker='.',label='7')
# plt.legend()
# ax8.plot(f.grid,eigenvalues[0,:],linestyle='--',marker='.',label='8')
# plt.legend()
# plt.show()


    ## SPECIES PLOT
    # manual_spec = ['CH', 'OH', 'H', 'HCO', 'H2O2']
    # manual_spec = ['CO', 'HCO', 'O', 'O2', 'C']
    # plt.figure()
    # for spec in manual_spec:
    #   y_spec = f.Y[gas.species_index(spec),:]
    #   plt.semilogy(f.grid,y_spec, label=spec)



    ## EIG FIGURE ## 

    # PLOT all eigenvalues stored
plt.figure()
for i in range(0,len(eigenvalues[:,0])):
    plt.plot(f.grid,eigenvalues[i,:],linestyle='--',marker='.',label=str(i))

plt.plot(f.grid,eig_track[:],linestyle='--',marker='.',label='patched')

plt.legend()
plt.xlabel('1D flame domain')
plt.title('Maximum eigenvalues (from lowest to highest)')
plt.show()

    # if EI_mode == 'debug':
        
    #   fig, ax = plt.subplots(2,1) 
    #   plt.subplot(2,1,1)
    #   for i in range(len(track_species)):
            
    #       plt.plot(np.array(tt)*1e6,expl_indices[track_species[i],:],linestyle='--', marker='o',label=EI_keys[track_species[i]])
    #   # plt.xlim((7.5e-5,9e-5))
    #   plt.xlabel('Residence time')
    #   plt.axvline(x=switch_time*1e6, ymin=0., ymax = 1, linewidth=1, color='k')
    #   # plt.xticks([7.5e-5, 8e-5, switch_time, 8.5e-5, 9e-5])

    #   plt.legend()
    #   # plt.show()

    #   plt.subplot(2,1,2)
    #   plt.plot(np.array(tt)*1e6,eigenvalues[order,:]/1e6,linestyle='--')
    #   plt.axvline(x=switch_time*1e6, ymin=0., ymax = 1, linewidth=1, color='k')

    #   titlefig = str(order) + '_provaEI' + '.pdf'
    #   plt.savefig(titlefig, bbox_inches='tight')

