# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 03:17:54 2023

@author: dell
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time
from mnn_core.maf_quad import *


def input_output_analysis_corr(mean1 = 1, std1 = 1, mean2 = 1, std2 = 1, ntrials = 1000):
    #inf = InteNFire(num_neurons = 2*ntrials)
    maf = MomentActivation()
    
    u = np.array([mean1,mean2])    
    s = np.array([std1,std2])
    rho = np.linspace(-1,1,31)
    
    #calculate analytical result
    maf_u = maf.mean(u,s)
    maf_s, _ = maf.std(u,s)
    maf_chi = maf.chi(u,s)
    maf_chi2 = maf.chi2(u,s)
    
    #!! CALCULATION is OFF ?????
    maf_rho = maf_chi[0]*maf_chi[1]*rho + 0.01*maf_chi2[0]*maf_chi2[1]*rho*rho
    
    return rho, maf_rho
    # SNN simulation - use existing results
    # T = min(10e3, 100/min(maf_u)) # adaptive simulation time
    # print('Using simulation time (ms):',T)
    # #simulation result of IF neuron
    # u = np.tile(u,ntrials)
    # s = np.tile(s,ntrials)
    
    # output_rho = rho.copy()    
    # for i in range(len(rho)):
    #     #spk_count = np.zeros((ntrials,2))
    #     SpkTime, _, _ = inf.run(T = T, input_mean = u, input_std = s, input_corr = rho[i], input_type = 'bivariate_gaussian', ntrials = ntrials)        
    #     spk_count = np.array([ len(k) for k in SpkTime]).reshape((ntrials,2))
    #     print('Progress: i={}/{}'.format(i,len(rho)))
    #     output_rho[i] = np.corrcoef( spk_count[:,0], spk_count[:,1] )[0,1]
        
        
def batch_analysis_corr():
    '''Fix mean2=std2=1 and vary mean1 and st1'''
    #u = np.array([0, 0.5, 1]) #each run takes about 3.5 min
    #s = np.array([0.5, 3, 10])
    u = np.linspace(0,2,11)
    s = np.linspace(0,5,11)
    
    #rho_out = np.zeros((11,len(u),len(s)))
    rho_maf = np.zeros((31,len(u),len(s)))
    
    start_time = time.time()
    
    for i in range(len(u)):
        for j in range(len(s)):
            rho_in, rho_maf[:,i,j] = input_output_analysis_corr(mean1 = u[i], std1 = s[j], ntrials = 1000)
    
            print('Time elapsed: {} min'.format( (-start_time + time.time())/60 ) )    
    #plt.plot(rho_in, rho_out, '.')
    #plt.plot(rho_in, rho_maf)
    np.save('D:/MNN/moment_neural_network/runs/snn/validate_corr_4_quad',{'rho_in':rho_in,'rho_out':None,'rho_maf':rho_maf,'u':u,'s':s})
    return rho_in, rho_maf, u, s 

def batch_analysis_corr2():
    '''Let mean1=mean2 and std1=std2, vary both.'''
    u = np.linspace(0,2,11)
    s = np.linspace(0,2,11)
    
    #rho_out = np.zeros((11,len(u),len(s)))
    rho_maf = np.zeros((31,len(u),len(s)))
    
    start_time = time.time()
    
    for i in range(len(u)):
        for j in range(len(s)):
            rho_in, rho_maf[:,i,j] = input_output_analysis_corr(mean1 = u[i], mean2 = u[i], std1=s[j], std2 = s[j], ntrials = 1000)
    
            print('Time elapsed: {} min'.format( (-start_time + time.time())/60 ) )    
    #plt.plot(rho_in, rho_out, '.')
    #plt.plot(rho_in, rho_maf)
    np.save('D:/MNN/moment_neural_network/runs/snn/validate_corr_5_quad',{'rho_in':rho_in,'rho_out':None,'rho_maf':rho_maf,'u':u,'s':s})
    return rho_in, rho_maf, u, s          

if __name__=='__main__':
    batch_analysis_corr2()