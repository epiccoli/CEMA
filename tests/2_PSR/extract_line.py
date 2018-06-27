import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import pdb
import os 
import scipy.linalg as LA
import pyjacob
# imports for set_mixture_jet
import cantera as ct 
import time as t
from scipy.interpolate import griddata
from set_mixture_jet import *

def get_nearest(vect,value):
    abs_diff = np.zeros_like(np.array(vect))
    for i in xrange(len(vect)):

        abs_diff[i] = np.abs(vect[i] - value)
    return np.amin(abs_diff)
    # return vect[idx]

def max_eig_gas(gas):

    # input arg: gas cantera object
    # output arg: eigenvalue vector, left and right eigenvector matrices 

    T = gas.T
    P = gas.P
    
    # #setup the state vector
    y = np.zeros(gas.n_species)
    y[0] = T

    for i in range(1,gas.n_species):
        if gas.species_name(i) != 'N2':
            # print(gas.species_name(i))
            y[i] = gas.Y[i]
    # y[1:] = gas.Y[:-1]
    
    # #create a dydt vector
    dydt = np.zeros_like(y)
    pyjacob.py_dydt(0, P, y, dydt)

    #create a jacobian vector
    jac = np.zeros(gas.n_species * gas.n_species)


    #evaluate the Jacobian
    pyjacob.py_eval_jacobian(0, P, y, jac)

    jac = jac.reshape(gas.n_species,gas.n_species)

    # Solve eigenvalue PB > D: eigenvalues
    D, vl, vr = LA.eig(jac, left = True)

    D=D.real
    # vl=vl.real
    # vr=vr.real
    
    # introduced to get rid of zeros: careful!
    # D = np.delete(D,np.where(D==0.0))
    # cannot delete here, must be after (affects EI)


    # return D, vl, vr

    return np.amax(D)

def mm2inch(*tupl):
    inch = 25.4
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

def test_get_ai(gas):

    print get_ai_time(2e-4,gas)

# gas,_,_ = set_mixture_wagner(gas,0.08,0.8)
# test_get_ai(gas)

def interp_2D(points, values, xq, yq):
    grid_data = griddata(points, values, (xq, yq), method='cubic')
    return grid_data

def interpolate_col(subdomain_df, meshgrid):
    ''' 
    Important: in the interpolated data, order follows (x,y) indexing with column changing faster 
    eg: (x1, y1), (x1, y2), ..., (x1, yn)
        (x2, y1), (x2, y2), ..., (x2, yn)
        ...     , ...     , ..., ...
        (xn,y1) , ...     , ..., (xn, yn)
    '''
    coord_headers = ['x', 'y', 'z']
    
    n_dim = meshgrid.shape[0]     

    xq = meshgrid[0]
    yq = meshgrid[1]
    
    # if n_dim > 2:
    #     zq = meshgrid[2]
    # do something more 

    column_headers = subdomain_df.columns.values 
    # Trim the column headers corresponding to coordinates used 
    for i in xrange(len(coord_headers)):
        idx_coord = np.where(column_headers == coord_headers[i])
        column_headers = np.delete(column_headers,idx_coord)
    data_headers = list(column_headers)
    # not sure, del column headers if long numpy array (free up memory?)

    if n_dim == 2:
        coord_headers.remove('z')

    # Coordinates of subdomain points
    sample_2D_coord = subdomain_df[coord_headers].values
    
    interp_data = np.zeros((len(xq.ravel()),len(data_headers)+n_dim))
    start = t.time()
    for i in xrange(len(data_headers)):
        new_col = interp_2D(sample_2D_coord, subdomain_df[data_headers[i]], xq, yq)
        interp_data[:,i] = new_col.ravel()
    
    print('Took {:.6f} sec to interpolate all {:d} columns with {:d} points and {:d} queries'.format(t.time()-start,len(data_headers),subdomain_df.shape[0],len(xq.ravel())))

    interp_data[:,-n_dim] = xq.ravel()
    interp_data[:,-n_dim+1] = yq.ravel()

    data_headers.append('x')
    data_headers.append('y')

    interp_df = pd.DataFrame(interp_data,columns=data_headers)

    return interp_df


def get_ai_time(Tai_guess, gas0):

    if os.path.isfile('xmgrace.txt'):
            os.remove('xmgrace.txt')

    npoints = 50000
    ####   adapt timestep for each mixture 
    timestep = Tai_guess/1e3

    #####   Create the batch reactor
    react = ct.IdealGasConstPressureReactor(gas0)

    # Now create a reactor network consisting of the single batch reactor
    # Reason: the only way to advance reactors in time is through a network
    sim = ct.ReactorNet([react])

    #####   Initial simulation time
    time = 0.0

    #####  parameters
    tim =np.zeros(npoints,'d')
    temp =np.zeros(npoints,'d')
    press =np.zeros(npoints,'d')
    enth =np.zeros(npoints,'d')

    ## INSTEAD OF FOR; while loop:
    tim = [0]
    temp = [gas0.T]
    dT = []
    
    peaked_grad = False
    increasing = True
    max_iter = False
    n=0
    while increasing == True:
        time += timestep
        sim.advance(time)
        tim.append(time)
    #       temp[n]= react.temperature() # old syntax
        temp.append(react.T)
        # enth[n]= react.thermo.enthalpy_mass
    #       test[n]= react.Y(fuel_species)
    #       press[n]= react.pressure()
        # press[n]= react.thermo.P
        # csv_append([sim.time, react.T, react.thermo.P],'xmgrace.txt','ab')

        #file1.write( '%10.3e %10.6f\n' % (sim.time(),react.enthalpy_mass()))
    #               file2.write( '%10.3e %10.6f %10.6f %10.6f %10.6f \n' % (sim.time,react.moleFraction(fuel_species),react.moleFraction('CO2'),react.moleFraction('H2O'),react.moleFraction('N2')))


        # update gradient
        dT.append(temp[-1] - temp[-2])

        if peaked_grad == False:
            try:
                if (dT[-1] < dT[-2]) and (dT[-1] > 0):
                    print('Temperature seems to have peaked at t={:.5e}'.format(tim[-1]))
                    # DEBUG: show temperature evolution up to inflection point found 
                    # plt.figure()
                    # plt.plot(tim,temp)
                    # plt.show()
                    peaked_grad = True
            except IndexError:
                pass
            # if peaked_grad == True:

        if peaked_grad == True and dT[-1] < 1e-6:

            print('passed threshold at t={:.6e}'.format(tim[-1]))
            increasing = False
        n += 1 

        # if max_iter == False:            
        #     if n > 8e4 and peaked_grad == True:
        #         print('passed number of iter at t={:.6f}'.format(tim[-1]))
        #         max_iter = True
        
        if n > 1e5:
            print('passed max_iter at t={:.6e}'.format(tim[-1]))
            break

    # DEBUG: show temperature evolution up to PSR stopped
    # plt.figure()
    # plt.plot(tim,temp)
    # plt.show()
            
    ########### calculate autoignition time
    Time = np.array(tim)
    Temperature=np.array(temp)

    #   enthalpy = np.array(enth)
    #   HR = np.diff(enthalpy)
    dt = np.diff(Time)

    #   HR = HR / dt
    #Tau_e = Time[HR==Hreact.max()][0] - Time[HR>0.05e0*Hreact.max()][0]

    dTdt = np.diff(Temperature)
    dTdt_max=dTdt.max()
    dTdt_max_index=np.where(dTdt==dTdt_max)[0][0]
    # 
    t_AI=Time[dTdt_max_index]

    return t_AI

#################### END FUNCTIONS #####################

Dj = 9.53e-3
Y_vect = np.arange(1.4,3.9,0.1)*Dj



# file_in = '../csv_files/try_line_through%20kernel.csv'
file_in = '../csv_files/slice_ScarDiss_Z_allSpec.csv'
file_interp = '../csv_files/slice_ScarDiss_Z_allSpec_INTERP.csv'


### PARAMETERS FOR SUBDOMAIN SELECTION ###
left = -0.6
right = 0
top = 4
bottom = 0
### Create interpolation grid ###
step_size = Dj/200       # fit 10 points in pipe outlet
# xq, yq = np.mgrid[left*Dj:right*Dj:step_size, bottom*Dj:top*Dj:step_size]
marg = step_size*5
# meshgrid = np.mgrid[left*Dj + marg : right*Dj - marg : step_size, bottom*Dj + marg : top*Dj - marg : step_size]

meshgrid = np.mgrid[left*Dj + marg : right*Dj - marg : step_size, bottom*Dj + marg : top*Dj - marg : step_size]

stop_y = meshgrid[1][0][-1]     # highest interpolated y value (last item in y coordinates = meshgrid[1][any point])
Y_vect = np.arange(step_size*20,stop_y,step_size*20)

if os.path.isfile(file_interp):

    interp_df = pd.read_csv(file_interp)

else: 

    df = pd.read_csv(file_in)
    # df = df[df['Ksi'] > 1e-3]
    df.rename(columns={'Points:0': 'x', 'Points:1': 'y', 'Points:2': 'z'}, inplace=True)      # essential to work with interpolate_col()
    df = df.query('x >= {:.7e} & x <= {:.7e} & y >= {:.7e} & y <= {:.7e}'.format(left*Dj,right*Dj,bottom*Dj,top*Dj))
    
    interp_df = interpolate_col(df, meshgrid)
    
    interp_df.to_csv(file_interp, index=False)


for line_y in Y_vect:
    file_out = './extracted_data/scar-diss_yD={:.2f}.csv'.format(line_y/Dj)
    
    if not os.path.isfile(file_out):
        
        # pdb.set_trace()
        # y_coords = interp_df['y'].values
        # nearest_y = get_nearest(y_coords,line_y)
        line_df = interp_df.query('x >= {:.7e} & x <= {:.7e} & y > {:.7f} & y < {:.7f} & Ksi > 0.001 & Ksi < 0.95'.format(left*Dj,right*Dj, line_y-step_size/4, line_y+step_size/4))
        line_df = pd.DataFrame(line_df)
        Z = line_df['Ksi'].values

        x = line_df['x'].values
            
        T = line_df['T'].values
        Xsc = line_df['ScarDiss']
        
        # hr = line_df['HR'].values#[idx]

        T_pure_mix = np.zeros_like(Z)
        t_AI_pure = np.zeros_like(Z)
        t_AI_real = np.zeros_like(Z)

        eig_pure = np.zeros_like(Z)
        eig_real = np.zeros_like(Z)

        gas_pure_mix = ct.Solution('Skeletal29_N.cti')
        gas_real_mix = ct.Solution('Skeletal29_N.cti')
        phi_max = 1.2



        AI_prev_pure = 1e-3
        AI_prev_real = 1e-3
        for i, Zi in enumerate(Z):
            
            gas_pure_mix, _, T_pure_mix[i] = set_mixture_wagner(gas_pure_mix,Zi,phi_max)
            gas_real_mix, _, _ = set_mixture_wagner(gas_real_mix,Zi,phi_max)
            # Overwrite Temperature for "real" mix conditions
            gas_real_mix.TP = T[i], 1.01325e5

            # EIGENVALUES
            eig_pure[i] = max_eig_gas(gas_pure_mix)
            eig_real[i] = max_eig_gas(gas_real_mix)

            # pdb.set_trace()
            # simulate pure mixing properties
            t_AI_pure[i] = get_ai_time(AI_prev_pure, gas_pure_mix)
            # simulate Z and T real 
            

            t_AI_real[i] = get_ai_time(AI_prev_real, gas_real_mix)

            AI_prev_pure = t_AI_pure[i]
            AI_prev_real = t_AI_real[i]

        line_df['t_AI_real'] = t_AI_real
        line_df['t_AI_pure'] = t_AI_pure
        line_df['T_pure'] = T_pure_mix
        line_df['eig_pure'] = eig_pure
        line_df['eig_real'] = eig_real
        line_df['ScarDiss'] = Xsc

        line_df.to_csv(file_out, index=False)
          

    line_df = pd.read_csv(file_out)

    x = line_df['x'].values / 9.53e-3
    Z = line_df['Ksi'].values
    T_real = line_df['T'].values
    T_pure = line_df['T_pure'].values
    t_AI_real = line_df['t_AI_real'].values
    t_AI_pure = line_df['t_AI_pure'].values

    eig_real = line_df['eig_real'].values
    eig_pure = line_df['eig_pure'].values
    Xsc = line_df['ScarDiss'].values
        
    # t_AI_pure[t_AI_pure>2] = np.nan
    # t_AI_real[t_AI_real>2] = np.nan

    ############## FIGURE ##############
    # fig, ax = plt.subplots(figsize=(4,4))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=mm2inch(90*2,110*2))


    ax1.plot(x,T_real,label=r'real')
    ax1.plot(x,T_pure,label=r'pure')
    ax1.set_ylabel(r'Temperature')
    ax1.legend()

    ax1b = ax1.twinx()
    # ax1.plot(x,hr,label='hr')
    ax1b.plot(x,Z,color='k')
    ax1b.set_ylabel(r'$Z$ mixture fraction')




    ax2.semilogy(x,t_AI_real,label='real')
    ax2.semilogy(x,t_AI_pure,label='pure')
    ax2.legend()
    ax2.set_xlabel(r'$x/D$ ')
    ax2.set_ylabel(r'$\tau_{\textsc{AI}} $')
    ax2.legend(loc=2)

    ax2b = ax2.twinx()
    ax2b.semilogy(x,eig_real,label='real',linestyle='--')
    ax2b.semilogy(x,eig_pure,label='pure',linestyle='--')
    ax2b.set_yticks([1e0, 1e1, 1e2, 1e3, 1e4, 1e5])
    ax2b.set_ylabel(r'CEM')
    ax2b.legend(loc=1)

    ax3.plot(x,T_real,label=r'real')
    ax3.plot(x,T_pure,label=r'pure')
    ax3.legend(loc=2)

    ax3b = ax3.twinx()
    ax3b.plot(x,Xsc,label=r'scalar diss',color='k')
    ax3b.legend(loc=1)

    plt.tight_layout()
    plt.savefig('test_fig.pdf')
    plt.suptitle(r'yD = {:.3f}'.format(line_y/Dj))
    plt.savefig('./extracted_data/line_1D_yD={:.3f}.pdf'.format(line_y/Dj))
    # plt.show()

    ############## FIGURE ##############
