# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 14:17:29 2023

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

path = './runs/vary_ie_ratio_fix_randseed__no_corr_nov_27/'  # with corr


uext = meta_dat['uext_array']
ie_ratio = meta_dat['ie_ratio_array']

U0 = np.zeros(12500, len(ie_ratio))
S0 = U.copy()

dU

for j in range(len(ie_ratio)):
    dat = load_data(path, indx)        
    config = dat['config'].item()        
    NE = config['NE']
    NI = config['NI']        
    
    #save fixed point
    U0[:,j] = dat['mnn_mean'][:,-1]
    S0[:,j] = dat['mnn_std'][:,-1]
    
    
results = {'U0':U0,
'S0':S0,
}    

np.savez(path+'fixed_pts.npz', **results)
    