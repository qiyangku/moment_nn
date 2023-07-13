# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 15:50:28 2023

Use large recurrent MNN to investigate information coding

@author: dell
"""
from mnn_core.rec_mnn_simulator import RecurrentMNN
from mnn_core.preprocessing import gen_synaptic_weight, InputGenerator
import numpy as np
from matplotlib import pyplot as plt
import time

def gen_config(shared_noise_scale = 0.065, N=100, cov_type = 'uniform'): #generate config file
    
    tot_noise = 76.5e-3+5e-3

    config = {
    'Vth': 1, #mV, firing threshold, default 20
    'Tref': 5, #ms, refractory period, default 5
    'NE': int(0.8*N),
    'NI': int(0.2*N),
    'var_ind' : tot_noise*(1-shared_noise_scale), # per ms, independent noise strength      
    'var_shr' : tot_noise*shared_noise_scale, # per ms, shared noise strength     
    'wee':{'mean': 6.0, 'std': 2.0},
    'wei':{'mean': -9.5, 'std': 2.0},
    'wie':{'mean': 5.4, 'std': 2.0},    
    'wii':{'mean': -8.9, 'std': 2.0},    
    #'wie':{'mean': 5.9, 'std': 0.0},    
    #'wii':{'mean': -9.4, 'std': 0.0},        
    'conn_prob': 0.2, #connection probability; N.B. high prob leads to poor match between mnn and snn
    'sparse_weight': False, #use sparse weight matrix; not necessarily faster but saves memory
    'randseed':0,
    'dT': 200, #ms spike count time window
    'cov_type': 'cosine',
    'mean_type': 'linear',
    }

    return config


def linear_fisher_info(num_neurons, cov_type, shared_noise_scale = 0.065):
    #initialize network
    # for this setting
    # shared_noise_scale is literally just the input corr coef
    
    config = gen_config(shared_noise_scale, N=num_neurons)    
    W = gen_synaptic_weight(config)
    input_gen = InputGenerator(config)
    input_gen.update(cov_type = cov_type) #update covariance matrix setting
    
    mnn_model = RecurrentMNN(config, W, input_gen)
    
    
    
    #calculate cov
    #stim = 0.1825
    #ds = 0.0025 # small change in the stimulus
    
    stim = 0.1825/2
    ds = 0.0025/2
    
    T_mnn = 20
    u,s,rho = mnn_model.run(T_mnn,stim, record_ts = False)
    #_,s,rho = rm.run(20)
    
    C = rho*s*s.T
    
    #calculate derivative of tuning function
    # re-instantiate input_gen otherwise it breaks!
    
    
    u1 ,_,_ = mnn_model.run(T_mnn,stim-ds, record_ts = False)    
    u2 ,_,_ = mnn_model.run(T_mnn,stim+ds, record_ts = False)        
    du = (u2-u1)/ds/2 #centered difference
    
    #du = np.ones_like(u)
    
    #calculate linear fisher info
    Cinv = np.linalg.pinv(C) #for dealing with degenerate C
    LFI = du.T.dot(Cinv).dot(du)
    
    
    #indx = np.triu_indices( num_neurons, k=1 )
    #rho[indx].mean() # E-E correlation
        
    return LFI, u,s,rho, input_gen


def run(cov_type, save_results=False):
    # check saturation of Fisher info
    # reproduction of Fig.2b in Moreno-Bote
    
    m = 31  # number of different network size
    m2 = 5  # number of different input corr
    
    #N = 5*np.round(np.logspace(1,2,m)) # forget log-space; it makes division by 5 really hard
    
    N = 5*np.round(np.linspace(0,200,m)) #must be multiples of 5 (to have integer NE, NI); took 4 min to run m=21
    #N = 5*np.round(np.linspace(0,40,m))
    LFI = np.zeros((m, m2))
    
    tmp = int(N[-1])
    output_mean = np.zeros((tmp,m2)) #mean firing rate
    output_std = np.zeros((tmp, m2))
    output_corr = np.zeros((tmp,tmp, m2))    
    input_cov = np.zeros((tmp,tmp,m2))
    
    if cov_type == 'uniform':
        shr_noise_lvl = 0.1*np.arange(m2)
    elif cov_type =='cosine':
        shr_noise_lvl = 0.2*np.arange(m2)
    
    t0 = time.time()
    for i in range(len(N)):
        if int(N[i])==0: # skip if network size is 0
            continue            
        print('Processing iteration... ',i)
        for j in range(len(shr_noise_lvl)):        
            LFI[i,j], u, s, rho, input_gen = linear_fisher_info( int(N[i]), cov_type, shr_noise_lvl[j] )
            if i==(len(N)-1): #save result of largest network only
                output_mean[:,j] = u.flatten()
                output_std[:,j] = s.flatten()
                output_corr[:,:,j] = rho
                input_cov[:,:,j] = input_gen.input_cov
            print('Time elapsed: ', np.round((time.time()-t0)/60,2))

    if save_results:
        np.savez('fisher_info_vs_N_cov_'+cov_type+'.npz', N=N, LFI=LFI, \
                 output_mean=output_mean, output_std=output_std, input_cov=input_cov, output_corr=output_corr, frac_shr_noise=shr_noise_lvl)
    return N, LFI, shr_noise_lvl, output_mean, output_std, output_corr, input_cov
    
    
if __name__=='__main__':   

    N, LFI, shr_noise_lvl, output_mean, output_std, output_corr, input_cov = run('uniform', save_results=True)    # no saturation (coz negative corr)
    N, LFI, shr_noise_lvl, output_mean, output_std, output_corr, input_cov = run('cosine', save_results=True)    # no saturation (coz negative corr)

    plt.close('all')
    plt.figure(figsize=(3.5,3))
    plt.loglog(N,LFI)
    plt.ylabel('Information rate (per ms)')
    plt.xlabel('Population size')
    plt.legend( [r'$\rho$ = {}'.format(k) for k in np.round(shr_noise_lvl,1)  ])
    plt.tight_layout()