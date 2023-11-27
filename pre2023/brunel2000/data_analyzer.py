# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 16:59:18 2023

@author: dell
"""

import numpy as np
from matplotlib import pyplot as plt
import os
import fnmatch

def load_data(path, indx=None):    
    data_files = os.listdir(path)    
    if indx == None:
        meta_dat = np.load(path+'meta_data.npz', allow_pickle=True)
        return meta_dat        
    for f in data_files:
        if fnmatch.fnmatch( f, str(indx).zfill(3) + '*.npz'):
            dat = np.load(path+f, allow_pickle=True)
            return dat

#path = './runs/pre2023_brunel_delay_2023_oct_13/'    # has oscillation
#path = './runs/pre2023_brunel_delay_zero_2023_oct_14/'    # no oscillation
#path = './runs/pre2023_brunel_delay_longer_2023_oct_14/' #delay = 0.5*tau, has oscillation
#path = './runs/pre2023_brunel_delay_03_2023_oct_14/'  #no oscillation
#path = './runs/pre2023_brunel_delay_05_slice_fine_2023_oct_16/'  #no oscillation
#path = './runs/pre2023_small_network_with_corr_slice_fine_2023_nov_21/'  # with corr
#path = './runs/pre2023_small_network_with_corr_fine_2023_nov_21/'  # with corr
path = './runs/vary_network_size_no_corr_nov_27/'  # with corr

meta_dat = load_data(path) #load meta data

try:
    uext = meta_dat['uext_array']
    ie_ratio = meta_dat['ie_ratio_array']
    N = []
    size = (len(uext), len(ie_ratio))
except:
    pass

try:
    uext = meta_dat['uext_array']
    N = meta_dat['N_array']    
    ie_ratio = []
    size = (len(uext), len(N))
except:
    pass


# [:,:,0] is ext; [:,:,1] is inh
mean_pop_avg = np.zeros(size+(2,))
ff_pop_avg = np.zeros(size+(2,))
corr_pop_avg = np.zeros(size+(3,))

mean_pop_std = np.zeros(size+(2,))
ff_pop_std = np.zeros(size+(2,))
corr_pop_std = np.zeros(size+(3,))

mean_quartiles = np.zeros(size+(2,))
ff_quartiles = np.zeros(size+(2,))

osc_amp = np.zeros(size)
osc_amp_ff = np.zeros(size)
osc_freq = np.zeros(size)

for i in range(size[0]):
    print('Processing... {}/{}'.format(i, size[0]))
    for j in range(size[1]):
        indx = np.ravel_multi_index((i,j), size )
        dat = load_data(path, indx)        
        config = dat['config'].item()        
        NE = config['NE']
        NI = config['NI']        
        
        u = dat['mnn_mean']
        s = dat['mnn_std']
        ff = s**2/u
        
        try:
            rho_ee = dat['mnn_corr'][:NE,:NE]
            rho_ei = dat['mnn_corr'][:NE,NE:]
            rho_ii = dat['mnn_corr'][NE:,NE:]
            
            rho_ee = rho_ee[np.triu_indices( NE ,k=1)] #upper triangle entries
            corr_pop_avg[i,j,0] = np.mean(rho_ee)
            corr_pop_std[i,j,0] = np.std(rho_ee)
            
            rho_ii = rho_ii[np.triu_indices( NI ,k=1)] #upper triangle entries
            corr_pop_avg[i,j,1] = np.mean(rho_ii)
            corr_pop_std[i,j,1] = np.std(rho_ii)
            
            corr_pop_avg[i,j,2] = np.mean(rho_ei)
            corr_pop_std[i,j,2] = np.std(rho_ei)
            
        except:
            corr_pop_avg[i,j,:] = None
            corr_pop_std[i,j,:] = None
        
        # average over second half of simulation, to deal with oscillating solutions
        cut_off = int(u.shape[1]/2) #discard the first half of time series
        u_time_avg = np.mean(u[:, cut_off:], axis = 1)  #average over time
        ff_time_avg = np.mean(ff[:, cut_off:], axis = 1) 
               
        # population stats
        mean_pop_avg[i,j,0] = np.mean(u_time_avg[:NE])
        mean_pop_avg[i,j,1] = np.mean(u_time_avg[NE:])
        ff_pop_avg[i,j,0] = np.mean(ff_time_avg[:NE])
        ff_pop_avg[i,j,1] = np.mean(ff_time_avg[NE:])
        
        mean_pop_std[i,j,0] = np.std(u_time_avg[:NE])
        mean_pop_std[i,j,1] = np.std(u_time_avg[NE:])
        ff_pop_std[i,j,0] = np.std(ff_time_avg[:NE])
        ff_pop_std[i,j,1] = np.std(ff_time_avg[NE:])
        
        mean_quartiles[i,j,:] = np.percentile(u_time_avg, [25,75])        
        ff_quartiles[i,j,:] = np.percentile(ff_time_avg, [25,75])
        
        # detect oscillation
        tmp = np.mean(u[:, cut_off:], axis=0) #population average, no time average
        tmp_ff = np.mean(ff[:, cut_off:], axis=0)
        
        
        #if ie_ratio[j]>4: # no oscilation found for excitation dominant regime
        osc_amp[i,j] = 0.5*(np.max(tmp)-np.min(tmp)) #rough estimate of oscillation amplitude
        osc_amp_ff[i,j] = 0.5*(np.max(tmp_ff)-np.min(tmp_ff))
        #if osc_amp[i,j]>1e-5:
        psd = np.abs(np.fft.fft(tmp))
        psd[0]=0
        psd = psd[:int(len(psd)/2)] #discard mirrored result
        osc_freq[i,j] = np.argmax(psd)/(config['T_mnn']/2*0.02)  # psd peak index * df, which is 1/simulation time (0.02 s is mem constant)
            

dat = {'ie_ratio':ie_ratio,
'uext':uext,
'N':N,
'mean_pop_avg':mean_pop_avg,
'ff_pop_avg':ff_pop_avg,
'mean_pop_std':mean_pop_std,
'ff_pop_std':ff_pop_std,
'osc_amp':osc_amp,
'osc_freq':osc_freq,
'osc_amp_ff':osc_amp_ff,
'mean_quartiles':mean_quartiles,
'ff_quartiles':ff_quartiles,
'corr_pop_avg':corr_pop_avg,
'corr_pop_std':corr_pop_std,
}

np.savez(path+'post_analysis.npz', **dat)
