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
            dat = np.load(path+f)
            return dat

#path = './runs/pre2023_brunel_delay_2023_oct_13/'    # has oscillation
#path = './runs/pre2023_brunel_delay_zero_2023_oct_14/'    # no oscillation
#path = './runs/pre2023_brunel_delay_longer_2023_oct_14/' #delay = 0.5*tau, has oscillation
#path = './runs/pre2023_brunel_delay_03_2023_oct_14/'  #no oscillation
path = './runs/pre2023_brunel_delay_05_fine_2023_oct_14/'  #no oscillation

meta_dat = load_data(path) #load meta data

uext = meta_dat['uext_array']
ie_ratio = meta_dat['ie_ratio_array']

# analysis 
size = (len(uext), len(ie_ratio))
mean_pop_avg = np.zeros(size)
ff_pop_avg = np.zeros(size)
mean_pop_std = np.zeros(size)
ff_pop_std = np.zeros(size)

osc_amp = np.zeros(size)
osc_freq = np.zeros(size)

for i in range(size[0]):
    print('Processing... {}/{}'.format(i, size[0]))
    for j in range(size[1]):
        indx = np.ravel_multi_index((i,j), size )
        dat = load_data(path, indx)
        u = dat['mnn_mean']
        s = dat['mnn_std']
        ff = s**2/u
        
        # average over second half of simulation, to deal with oscillating solutions
        cut_off = int(u.shape[1]/2)
        u_time_avg = np.mean(u[:, cut_off:], axis = 1)  #average over time
        ff_time_avg = np.mean(ff[:, cut_off:], axis = 1) 
               
        # population stats
        mean_pop_avg[i,j] = np.mean(u_time_avg)
        ff_pop_avg[i,j] = np.mean(ff_time_avg)
        mean_pop_std[i,j] = np.std(u_time_avg)
        ff_pop_std[i,j] = np.std(ff_time_avg)
        
        # detect oscillation
        tmp = np.mean(u[:, cut_off:], axis=0) #population average, no time average
                
        if ie_ratio[j]>4: # no oscilation found for excitation dominant regime
            osc_amp[i,j] = 0.5*(np.max(tmp)-np.min(tmp)) #rough estimate of oscillation amplitude
            
            #if osc_amp[i,j]>1e-5:
            psd = np.abs(np.fft.fft(tmp))
            psd[0]=0
            psd = psd[:int(len(psd)/2)] #discard mirrored result
            osc_freq[i,j] = np.argmax(psd)/(config['T_mnn']*0.02)  # psd peak index * df, which is 1/simulation time (ms)
            

dat = {'ie_ratio':ie_ratio,
'uext':uext,
'mean_pop_avg':mean_pop_avg,
'ff_pop_avg':ff_pop_avg,
'mean_pop_std':mean_pop_std,
'ff_pop_std':ff_pop_std,
'osc_amp':osc_amp,
'osc_freq':osc_freq,
}

np.savez(path+'post_analysis.npz', **dat)
