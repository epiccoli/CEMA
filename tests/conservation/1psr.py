import sys
import pandas as pd

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

def build_conservative_basis():
    # row vectors of conservative modes for H2 li2003
    B = np.array([[1,0,0,0,0,0,0,0,0,0,0,0,0],
              [0,1,0,0,0,0,0,0,0,0,0,0,0],
              [0,0,1,0,0,0,0,0,0,0,0,0,0],
              [0,0,0,1,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,1,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,1,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,1,0],
              [0,0,0,0,0,0,0,0,0,0,0,0,1]])
    inv = LA.inv(B.dot(B.T))
    Bc = B.T.dot(inv).dot(B)        # basis of conservative modes 

    return Bc


def defect(Bc, be):
    be = be/LA.norm(be) # normalise (just in case)    
    ba = (be.T).dot(Bc) # best approximation of be spanned by conservative modes basis

    defectiveness = be.dot(ba)

    return defectiveness

def find_ind(vect, low_bound, high_bound):

    idx = np.zeros_like(vect)

    for i in range(len(vect)):

        if vect[i] > low_bound and vect[i] < high_bound:
            idx[i] = 0

        else:
            idx[i] = 1

    return idx

 
def stats(vector,keys):

    max_val = np.amax(vector)
    max_idx = np.argmax(vector)

    print "Max value: ", max_val, " at position ", max_idx, " -> ", keys[max_idx]


# Gas properties
phi = 1.0
P = 101325
T = 1100
fuel_spec = 'H2'

# Simulation conditions
npoints = 1000
timestep = 1.5e-7
CEMA_interval = 1 # only divisors of npoints

N_eig = 8
N_EI = 1
#### options 
first_eigenmode = True
first_ei = True



# Create gas object
gas = ct.Solution('Li_2003.cti')


## REORDER CANTERA SPECIES AS PYJAC WOULD:
specs = gas.species()[:]
N2_ind = gas.species_index('AR')
gas = ct.Solution(thermo='IdealGas', kinetics='GasKinetics',
        species=specs[:N2_ind] + specs[N2_ind + 1:] + [specs[N2_ind]],
        reactions=gas.reactions())
# print gas.species_name(28) # >> should give N2



EI_keys = ['']*gas.n_species

EI_keys[0] = 'T'
for i in range(1,gas.n_species):
    EI_keys[i] = gas.species_name(i-1)
    print gas.species_name(i)
print EI_keys   


# prova manuale : H (entry 1)

## SET EQUIVALENCE RATIO TO phi, temperature and pressure
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

# with open('xmgrace.txt','w') as file1:
file1 = open('xmgrace.txt','w')



val_ei = []
lambda_patched1 = []
lambda_patched2 = []

eigenvalues = np.zeros((N_eig,npoints))
expl_indices = np.zeros((gas.n_species,npoints))
CEM = np.zeros(npoints)


H2_cons_eig = np.empty(npoints)

track_species = []

count=0

most_aligned_eig = []
most_aligned_ei = []
alignment = np.zeros(gas.n_species)




# df=pd.read_csv('ei_max_H2.csv', sep=',',header=None)
# EI_max = df.values.ravel()


start = t.time()

Bc = build_conservative_basis()



for n in range(npoints):
    time += timestep
    sim.advance(time)
    
    tim[n] = time
    temp[n]= r.T

    D, L, R = solve_eig_gas(gas)    

    # TRY1:
    # conservative_pos = [1,2,3,9,10,11,12]
    # non_conservative_pos = [0,4,5,6,7,8]
    # conservative_mask = np.empty(len(D))
    # for spec in range(len(D)):

    #   ei_curr = EI(D,L,R,spec)

    #   for j in conservative_pos:

    #       if ei_curr[j] > 0.9:

    #           conservative_mask[spec] = 1

    # non_conservative_modes = np.ma.array(D,mask=conservative_mask)

    # CEM[n] = np.amax(non_conservative_modes)

    # TRY2:
    non_conservative_modes = np.array([[1,0,0,0,0,0,0,0,0,0,0,0,0],
                                        [0,1,0,0,0,0,0,0,0,0,0,0,0],
                                        [0,0,1,0,0,0,0,0,0,0,0,0,0],
                                        [0,0,0,1,0,0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0,0,1,0,0,0],
                                        [0,0,0,0,0,0,0,0,0,0,1,0,0],
                                        [0,0,0,0,0,0,0,0,0,0,0,1,0],
                                        [0,0,0,0,0,0,0,0,0,0,0,0,1]])

    temperature_cons = np.array([1,0,0,0,0,0,0,0,0,0,0,0,0])
    temperature_cons = np.array([0,1,0,0,0,0,0,0,0,0,0,0,0])


    alignment = np.zeros(len(D))
    count = 0
    for i in range(len(D)):

        ei_curr = EI(D,L,R,i)
        alignment[i] = parallelism(ei_curr,temperature_cons)

        if alignment[i] > 0.99:
            count += 1 

    # print(np.sort(alignment))
    H2_cons_eig[n] = D[np.argsort(alignment)[-1]]
    # print(D[np.argsort(alignment)[-1]])
    # pdb.set_trace()
    print count

    



    # if n%100 == 0:
    
    #     print(time)
    #     for i in range(len(D)):
    #         if D[i] < 1 and D[i] > -1:
    #             print D[i]
    #             ei_curr = EI(D,L,R,i)
    #             print ei_curr
    #             pdb.set_trace()
            # else:
                # ei_curr = EI(D,L,R,i)
                # print ei_curr[-4:]
            # L_curr = L[:,i]
                # R_curr = R[:,i]

                # print('Eigenvalue {:d} => {:.3f}'.format(i+1,D[i]))
                # print(L_curr)#/LA.norm(L_curr))
                # print(R_curr)
                # print(ei_curr)

                # for mode in range(len(non_conservative_modes[:,0])):
                # print(parallelism(ei_curr,non_conservative_modes[mode,:]))
                # pdb.set_trace()

            # for mode in range(len(non_conservative_modes[:,0])):
            #   print(parallelism(ei_curr,non_conservative_modes[mode,:]))
    

    count = 0
    for i in range(len(D)):
        if D[i] == 0.0:
            count += 1

        # if D[i]<1 and D[i]> -1:
        #     count +=1

    if count != 4:
        print('This point is different')
        pdb.set_trace()



    eigenvalues[:,n] = D[np.argsort(D)[-N_eig:]]


    # explore conservative modes EI
    # if n%20 == 0:
    #   mask_big = find_ind(D,-1,1)
    #   mask_small = np.abs(mask_big-1)
        
        
    #   small_eig = np.ma.array(D,mask=mask_big)
    #   small_idx = np.ma.array(np.arange(0,len(D)),mask=mask_big)

    #   big_eig = np.ma.array(D,mask=mask_small)
    #   big_idx = np.ma.array(np.arange(0,len(D)),mask=mask_small)

    #   # pdb.set_trace()
    #   print 'Time : ', time

    #   for i in big_idx.compressed():

    #       # print(EI(D,L,R,i))
    #       ei = EI(D,L,R,i)
    #       for j in conservative_pos:

    #           if ei[j]>0.5:
    #               pdb.set_trace()
    #               print(ei)



    #   for i in small_idx.compressed():

    #       ei = EI(D,L,R,i)
    #       for j in non_conservative_pos:

    #           if ei[j]>0.5:
    #               print('conservative mode has high EI for non conservative spec')
    #               pdb.set_trace()
    #               print(D[i])
    #               print(ei)

        
    # for i in range(len(D)):

    #   alignment[i] = parallelism(EI(D,L,R,i), EI_max)

    # CE[n] = D[np.argmax(alignment)]



    
    ## MANUALLY CHANGE WHICH EIG on which to base EI calc: -6 is eig 1
    #   7 6 5 4 3 2 1 0         i
    # -[1 2 3 4 5 6 7 8] 
    switch_time = 8.249e-5
    # if tt[count] < switch_time:
    #   max_idx = np.argsort(D)[-1]
    #   lambda_patched1.append(D[max_idx])
    # else:
    #   max_idx = np.argsort(D)[-8]
    #   lambda_patched2.append(D[max_idx])


    # WHEN LOOKING AT WHICH EI IS WHICH (DEBUG EI FOR PLOTS)
    order=-1
    max_idx = np.argsort(D)[order]
    

    expl_indices[:,n] = EI(D,L,R,max_idx)
    main_EI = np.argsort(expl_indices[:,n])[-N_EI:]


    # manual identification of important species indices
    # print main_EI
    # pdb.set_trace()


    track_species = np.union1d(main_EI,track_species)

    # stats(expl_indices,EI_keys)

    # track species 

    
    

    # enth[n]= r.thermo.enthalpy_mass
    press[n]= r.thermo.P
    




end = t.time()

pdb.set_trace()

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


track_entries = ['T', 'H', 'O', 'OH', 'HO2', 'O2']
idx_entries = [0, 4, 5, 6, 7, 2]

idx_entries = [0, 1, 2, 4]
track_entries = ['T', 'H2', 'O2', 'H']

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



plt.figure()
plt.plot(tim,H2_cons_eig/1e6)
plt.show()

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



if mode == 'selectedEig':   
    tt = tim
    tt1 = tt[np.where(tt<switch_time)]
    tt2 = tt[np.where(tt>switch_time)]

    fig = plt.figure()
    ax_big = fig.add_subplot(111, frameon=False)    # The big subplot
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')

    ax1 = fig.add_subplot(911)
    ax2 = fig.add_subplot(912)
    ax3 = fig.add_subplot(913)
    ax4 = fig.add_subplot(914)
    ax5 = fig.add_subplot(915)
    ax6 = fig.add_subplot(916)
    ax7 = fig.add_subplot(917)
    ax8 = fig.add_subplot(918)
    ax9 = fig.add_subplot(919)

    # ax_big.spines['top'].set_color('none')
    # ax_big.spines['bottom'].set_color('none')
    # ax_big.spines['left'].set_color('none')
    # ax_big.spines['right'].set_color('none')
    # ax_big.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

    # ax_big.set_xlabel(r'$\mathrm{Residence\;time}\quad \left[ \mu s^{-1} \right]$')
    # ax_big.set_ylabel(r'$\lambda_{expl} \quad \mu s ^{-1}$')
    # ax_big.set_xlabel('ciao')
    # ax_big.set_ylabel(r'$\lambda_{expl} \quad \mu s ^{-1}$')

    # PLOT ONLY FOR COMPARISON
    i=7
    plt.subplot(9,1,1)
    plt.plot(tt*1e6,eigenvalues[i,:]/1e6,linestyle='--', marker='.',color='r',label=r'$1^{st} \, \mathrm{eig}$') 
    plt.ylim((-0.3,0.4))
    plt.axvline(x=switch_time*1e6, linewidth=1, linestyle='--', color='k')
    plt.legend(loc='right')
    
    ax1.axes.get_xaxis().set_visible(False)
    plt.yticks([-0.2, 0.0, 0.2])
    
    i=0
    plt.subplot(9,1,2)
    plt.plot(tt*1e6,eigenvalues[i,:]/1e6,linestyle='--', marker='.',color='b',label=r'$8^{th} \, \mathrm{eig}$')
    plt.ylim((-0.3,0.4))
    plt.axvline(x=switch_time*1e6, linewidth=1, linestyle='--', color='k')
    plt.legend(loc='right')
    
    ax2.axes.get_xaxis().set_visible(False)
    plt.yticks([-0.2, 0.0, 0.2])
    
    plt.subplot(9,1,3)
    plt.plot(tt1*1e6,np.array(lambda_patched1)/1e6,linestyle='--', marker='.', color='r', label=r'$\lambda_{explosive}$')
    plt.plot(tt2*1e6,np.array(lambda_patched2)/1e6,linestyle='--', marker='.', color='b',label=r'$\lambda_{explosive}$')
    plt.ylim((-0.3,0.4))
    plt.axvline(x=switch_time*1e6, linewidth=1, linestyle='--', color='k')
    plt.legend(loc='right')
    
    ax3.axes.get_xaxis().set_visible(False)
    plt.yticks([-0.2, 0.0, 0.2])

    i=6
    plt.subplot(9,1,4)
    plt.plot(tt*1e6,eigenvalues[i,:]/1e6,linestyle='--', marker='.',color='k',label=r'$2^{nd} \, \mathrm{eig}$')
    # plt.ylim((-0.2,0.4))
    plt.axvline(x=switch_time*1e6, linewidth=1, linestyle='--', color='k')
    plt.legend(loc='right')
    
    ax4.axes.get_xaxis().set_visible(False)

    i=5
    plt.subplot(9,1,5)
    plt.plot(tt*1e6,eigenvalues[i,:]/1e6,linestyle='--', marker='.',color='k',label=r'$3^{rd} \, \mathrm{eig}$')
    # plt.ylim((-0.2,0.4))
    plt.axvline(x=switch_time*1e6, linewidth=1, linestyle='--', color='k')
    plt.legend(loc='right')
    ax5.set_ylabel(r'$\lambda_{i} \, \left[ \mu s ^{-1}\right]$')
    ax5.axes.get_xaxis().set_visible(False)

    i=4
    plt.subplot(9,1,6)
    plt.plot(tt*1e6,eigenvalues[i,:]/1e6,linestyle='--', marker='.',color='k',label=r'$4^{th} \, \mathrm{eig}$')
    # plt.ylim((-0.2,0.4))
    plt.axvline(x=switch_time*1e6, linewidth=1, linestyle='--', color='k')
    plt.legend(loc='right')
    
    ax6.axes.get_xaxis().set_visible(False)

    i=3
    plt.subplot(9,1,7)
    plt.plot(tt*1e6,eigenvalues[i,:]/1e6,linestyle='--', marker='.',color='k',label=r'$5^{th} \, \mathrm{eig}$')
    # plt.ylim((-0.2,0.4))
    plt.axvline(x=switch_time*1e6, linewidth=1, linestyle='--', color='k')
    plt.legend(loc='right')
    
    ax7.axes.get_xaxis().set_visible(False)

    i=2
    plt.subplot(9,1,8)
    plt.plot(tt*1e6,eigenvalues[i,:]/1e6,linestyle='--', marker='.',color='k',label=r'$6^{th} \, \mathrm{eig}$')
    # plt.ylim((-0.2,0.4))
    plt.axvline(x=switch_time*1e6, linewidth=1, linestyle='--', color='k')
    plt.legend(loc='right')
    
    ax8.axes.get_xaxis().set_visible(False)

    i=1
    plt.subplot(9,1,9)
    plt.plot(tt*1e6,eigenvalues[i,:]/1e6,linestyle='--', marker='.',color='k',label=r'$7^{th} \, \mathrm{eig}$')
    # plt.ylim((-0.2,0.4))
    plt.axvline(x=switch_time*1e6, linewidth=1, linestyle='--', color='k')
    plt.legend(loc='right')


    # po
    # post_treatment of most_aligned_eig and tt
    # discard values of most_aligned_eig/1e6 < 0.5

    most_aligned_eig = np.array(most_aligned_eig)
    # pdb.set_trace()
    tt = np.delete(tt,np.where(most_aligned_eig < -8e5))
    most_aligned_eig = np.delete(most_aligned_eig,np.where(most_aligned_eig < -8e5))

    plt.xlabel(r'$\mathrm{Residence\;time}\quad \left[ \mu s \right]$')
    # plt.plot(tt,np.array(most_aligned_eig)/1e6,'.',label='most_aligned_eig')
    # plt.plot(tt,np.array(most_aligned_ei)/1e6,'.',label='most_aligned_ei')


    plt.suptitle(r'Time scale of the chemical explosive mode in PSR simulation of hydrogen-air at $\phi$=1.0, 1atm, initial temperature of 1100 K', fontsize=17)


    plt.show()


# print 'maximum heat release rate at time ', tim[np.argmax(np.diff(temp))]


plt.show()