# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 15:01:25 2023

For benchmarking speed of MA vs other methods.
No recurrent network is used.

@author: Yang Qi
"""

from mnn_core.maf import MomentActivation
from pre2023.FPE_solver import FPE_solver
from pre2023.look_up_solver import LookUpSolver
import numpy as np
import time
from scipy.integrate import quad

#grid size
n = 20 # don't need that many samples since average is taken
num_neurons = 5
num_trials = 100


input_mean = np.linspace(-10,10,n)
input_std = np.linspace(0,20,n)

dT = np.zeros((n,n,8)) #

frate = np.zeros((n,n,8))

ma = MomentActivation()
fpe = FPE_solver()
lookup = LookUpSolver()


ma2 = MomentActivation() #create another instance of moment activation

def quad_mean_map(ma2 , u,s):
    ub = (1-u)/(s*np.sqrt(ma2.L))
    lb = -u/(s*np.sqrt(ma2.L))
    G = ma2.ds1.int_brute_force(ub) - ma2.ds1.int_brute_force(lb)    
    G = G*2/0.05
    return 1/(5+G)

def quad_std_map(ma2 , u,s):
    ub = (1-u)/(s*np.sqrt(ma2.L))
    lb = -u/(s*np.sqrt(ma2.L))
    s = ma2.ds2.int_brute_force(ub) - ma2.ds2.int_brute_force(lb)    
    return

def g_quad(x):
    '''accept scalar input'''
    y,_ = quad(lambda x:  np.exp(-x**2)  , -np.inf, x)       
    return np.exp(x**2)*y

def quad_chi_map(u, s, u_cache):    
    ub = (1-u)/(s*np.sqrt(ma2.L))
    lb = -u/(s*np.sqrt(ma2.L))
    g = np.zeros(len(u))
    for i in range(len(g)):
        g[i] = g_quad(ub[i])-g_quad(lb[i])
    L = 0.05
    chi = (2/L/np.sqrt(L))*u_cache**2/s*g

indx = np.random.permutation(n**2) # shuffle the order of parameters

for kk in range(n**2):
    if kk % 100 ==0:
        print('{}/{}'.format(kk,n**2))
    i,j = np.unravel_index( indx[kk], (n,n))

    u = np.ones( num_neurons) * input_mean[i]
    s = np.ones( num_neurons) * input_std[j]
    
    # MA
    t0 = time.perf_counter()
    r = ma.mean( u,s)
    u_cache = r.copy()
    dT[j,i,0] = time.perf_counter() - t0 # MA with mean only
    frate[j,i,0] = np.mean(r)
    
    t0 = time.perf_counter()
    ma.std( u,s)
    dT[j,i,1] = time.perf_counter() - t0 # MA full
    
    t0 = time.perf_counter()
    ma.chi( u,s)
    dT[j,i,2] = time.perf_counter() - t0 # MA full
    
    # FPE    
    r,dt = fpe.run(u,s)  #this thing has a wrapper that auto generate time count
    dT[j,i,3] = np.mean(dt)
    frate[j,i,3] = np.mean(r)
    
    # Look-up table
    t0 = time.perf_counter()
    r = lookup.interp_mean( (u,s) )    
    dT[j,i,4] = time.perf_counter() - t0
    frate[j,i,4] = np.mean(r)
    
    # direct integration
    t0 = time.perf_counter()
    r = quad_mean_map(ma2, u,s )    
    dT[j,i,5] = time.perf_counter() - t0
    frate[j,i,5] = np.mean(r)
    
    t0 = time.perf_counter()
    quad_std_map(ma2, u,s )    
    dT[j,i,6] = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    quad_chi_map(u,s, u_cache)    
    dT[j,i,7] = time.perf_counter() - t0
    
    

dT = dT/num_neurons*1e6 #micro seconds

x = ['MA_u', 'MA_s', 'MA_x', 'FPE', 'Look-up', 'Quad_u', 'Quad_s','Quad_x']
np.savez('benchmark_speed_comparison.npz', input_mean=input_mean, dT=dT, n=n, num_neurons=num_neurons, input_std=input_std, label=x)




#%% box plot
from pre2023.utils import *
import matplotlib.pyplot as plt

dat = np.load('benchmark_speed_comparison.npz')
dT = dat['dT']
x = dat['label']

plt.close('all')
plt.figure(figsize=(3.5,3))

y = dT.reshape(dat['n']**2, 8)

z = {r'MA $\mu$': y[:,0], r'MA $\sigma$': y[:,1], r'MA $\chi$':y[:,2], r'DI $\mu$':y[:,5], \
           r'DI $\sigma$': y[:,6], r'DI $\chi$':y[:,7], 'FPE':y[:,3], 'Look-up':y[:,4]  }

plt.boxplot(z.values(), labels = z.keys(),  sym='' )
plt.yscale('log')
plt.ylabel(r'CPU time ($\mu s$)')
plt.xticks(rotation = 30) 
plt.tight_layout()

#%% table

dat = np.load('benchmark_speed_comparison.npz')
dT = dat['dT']
x = dat['label']
n = dat['n']
y =dT.reshape(n**2,8)

y_median = np.median(y,axis=0).round(2)
y_Q1 = np.quantile(y, 0.25, axis=0).round(2)
y_Q2 = np.quantile(y, 0.75, axis=0).round(2)

for i in range(8):
    print( '{}: {} ({},{})'.format(x[i], y_median[i], y_Q1[i], y_Q2[i] )  )

#y = np.mean(dT.reshape(n**2,8),axis=0)
#dy = np.std(dT.reshape(n**2,8),axis=0)
#x= np.arange(5)
# perhaps a table is better!!


#%% bar plot

dat = np.load('benchmark_speed_comparison.npz')
dT = dat['dT']
x = dat['label']
n = dat['n']
y =dT.reshape(n**2,8)

y_mean = np.mean(y,axis=0)
y_std = np.std(y,axis=0)

plt.close('all')
plt.figure(figsize=(3.5,3))
plt.bar(x, y_mean)

plt.errorbar(x, y_mean, yerr= y_std, fmt=".", color="k")

plt.yscale('log')
#plt.ylim(1e1,1e6)
plt.ylabel('CPU time ($\mu s$)')
plt.tight_layout()

   

