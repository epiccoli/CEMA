#!/usr/bin/env python

# script derived from example at: http://slackha.github.io/pyJac/examples.html 
import sys
sys.path.append("../0_pyjac_wrap_example/Li_2003/.")      # add Li_2003 pyjacob's path to system's path

import pyjacob                                  # import pyjacob (jacobian static library, precompiled before with the same mechanism file as the one used later by cantera, "Li_2003.cti")

import cantera as ct
import numpy as np


# create gas from original mechanism file Li_2003.cti 
gas = ct.Solution('Li_2003.cti')
# reorder the gas to match pyJac (set last_spec as the first between [N2, AR, HE] in order of appearance in the cti file 
last_spec = 'AR'
last_idx = gas.species_index(last_spec)

specs = gas.species()[:]
# rearrange species in solution object
gas = ct.Solution(thermo='IdealGas', kinetics='GasKinetics',
        species=specs[:last_idx] + specs[last_idx + 1:] + [specs[last_idx]],
        reactions=gas.reactions())

#set the gas state
T = 1200
P = ct.one_atm
gas.TPY = T, P, "CH4:0.2, O2:2, N2:7.52"

# #setup the thermochemical state vector
y = np.zeros(gas.n_species)
y[0] = T
y[1:] = gas.Y[:-1] # all species mass fractions except the one belonging to last_spec

# create a dy/dt vector
dydt = np.zeros_like(y)
pyjacob.py_dydt(0, P, y, dydt)

#create a jacobian vector
jac = np.zeros(gas.n_species * gas.n_species)

#evaluate the Jacobian
pyjacob.py_eval_jacobian(0, P, y, jac)

jac = jac.reshape(gas.n_species,gas.n_species)          # reshape into matrix for further calculations

print(jac) 

print('test: successful')
