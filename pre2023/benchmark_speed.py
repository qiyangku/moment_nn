# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 15:01:25 2023

For benchmarking speed of MA vs other methods.
No recurrent network is used.

@author: Yang Qi
"""



from mnn_core.maf import MomentActivation
import numpy as np
import time

#grid size
n = 100
num_neurons = 5
num_trials = 100


input_mean = np.linspace(-10,10,n)
input_std = np.linspace(0,20,n)

dT = np.zeros((n,n,3, num_trials))

ma = MomentActivation()

indx = np.random.permutation(n**2) # shuffle the order of parameters

for k in range(num_trials):
    print('{}/{}'.format(k,num_trials))
    for kk in range(n**2):
        i,j = np.unravel_index( indx[kk], (n,n))
#    for j in range(n):        
#        for i in range(n):        
        #for the same set of (u,s), must do mean, std, chi in sequence as cache is used
        u = np.ones( num_neurons) * input_mean[i]
        s = np.ones( num_neurons) * input_std[j]
        
        t0 = time.perf_counter()
        ma.mean( u,s)
        dT[j,i,0,k] = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        ma.std( u,s)
        dT[j,i,1,k] = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        ma.chi( u,s)
        dT[j,i,2,k] = time.perf_counter() - t0

dT = dT/num_neurons*1e6 #micro seconds

#np.savez('benchmark_speed.npz', input_mean=input_mean, dT=dT, n=n, num_neurons=num_neurons, input_std=input_std)



#%%
from pre2023.utils import *
import matplotlib.pyplot as plt

def plot_boundary(cut_off):
    # subthreshold regime    
    if cut_off>0:
        ubar = np.array([-10,1])
    else:
        ubar = np.array([1,10])
    sbar = (1 - ubar)/np.sqrt(0.05)/cut_off
    plt.plot(ubar,sbar,'k')
    
    

dat = np.load('benchmark_speed.npz')

input_mean = dat['input_mean']
input_std = dat['input_std']
dT = dat['dT']

plt.close('all')

# check the range of data
#tmp = np.median(dT[:,:,2,:], axis=-1)
#plt.hist(tmp.flatten(),50)
##%%

vmax = [46.5, 103.5, 8]
vmin = [44, 100, 7.6]

extent = [input_mean[0], input_mean[-1], input_std[0], input_std[-1]]
cmap = ['inferno','inferno','inferno']
title = ['Mean firing rate', 'Firing variability', 'Linear res. coef.']

for i in range(3):
    plt.figure(figsize=(4,3))
    img = np.median(dT[:,:,i,:], axis=-1)
    img = medianFilter(img) # remove shot noise
    plt.imshow( img, origin = 'lower', extent=extent, vmin = vmin[i], vmax = vmax[i], cmap =cmap[i]) #unit: ms
    
    # if i==0:        
    #     plot_boundary(10)
    #     plot_boundary(-10)
    #     plot_boundary(6)
    #     #plot_boundary(4.5)
    
    plt.colorbar(label = r'CPU time ($\mu s$)')
    plt.xlabel(r'Input current mean $\bar{\mu}$')
    plt.ylabel(r'Input current std $\bar{\sigma}$')    
    plt.title(title[i])
    plt.tight_layout()
    



