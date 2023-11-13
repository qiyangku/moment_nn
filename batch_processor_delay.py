# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 01:06:29 2021
A helper function for running multiple runs with different config
@author: dell
"""

from pre2023.brunel2000.rec_mnn_simulator import *
#from matplotlib import pyplot as plt
import numpy as np
import os, sys, time

#For PBS:
#INPUT: search_space a dictionary of lists
#       PBS_array index
#Wrapper: nested loop over the search_space
#Output the config dictionary

def gen_config(N=100, ie_ratio=4.0, uext=10.0): #generate config file
    w = 0.1    

    config = {
    'Vth': 20, #mV, firing threshold, default 20
    'Vres': 10, #mV reset potential; default 0
    'Tref': 2, #ms, refractory period, default 5
    'NE': int(0.8*N),
    'NI': int(0.2*N),
    'g': ie_ratio,     #I-E ratio
    'wee':{'mean': w, 'std': 0},
    'wei':{'mean': -w*ie_ratio, 'std': 0},
    'wie':{'mean': w, 'std': 0},    
    'wii':{'mean': -w*ie_ratio, 'std': 0},
    'uext': uext, # external firing rate kHz; rate*in-degree*weight = 0.01*1000*0.1 = 1 kHz
    #'wie':{'mean': 5.9, 'std': 0.0},    
    #'wii':{'mean': -9.4, 'std': 0.0},        
    'conn_prob': 0.1, #connection probability; N.B. high prob leads to poor match between mnn and snn
    'sparse_weight': False, #use sparse weight matrix; not necessarily faster but saves memory
    'randseed':None,
    'dT': 200, #ms spike count time window
    'delay': 0.1, # synaptic delay (uniform) in Brunel it's around 2 ms (relative to 20 ms mem time scale)
    'dt':0.02, # integration time step for mnn
    'T_mnn':20,
    }

    return config


def run(config, record_ts = True ):
    
    W = gen_synaptic_weight(config) #doesn't take too much time with 1e4 neurons    
    input_gen = InputGenerator(config)
    mnn_model = RecurrentMNN(config, W, input_gen)
    #snn_model = InteNFireRNN(config, W , input_gen)
    
    # simulate mnn
    t0 = time.perf_counter()
    u,s = mnn_model.run_no_corr(config['T_mnn'], record_ts = record_ts)
    print('Time elapsed (min): ', int(time.perf_counter()-t0)/60)
    
    return u,s
    
    
if __name__ == "__main__":    
    
    indx = int(sys.argv[1]) #use this to pick a particular config
    exp_id = sys.argv[2] # id for the set of experiment
    
    uext_array = np.linspace(0.0, 40.0 ,11)[1:]
    ie_ratio_array = np.linspace(0.0, 8.0 ,9)
    
    i,j = np.unravel_index(indx, [len(uext_array), len(ie_ratio_array)] ) 
    # to get linear index back from subscripts, use: np.ravel_multi_index((i,j),[len(uext_array), len(ie_ratio_array)])
    
    config = gen_config(N=12500, ie_ratio=ie_ratio_array[j], uext=uext_array[i])
    
    u,s = run(config)
    
    
    path =  './runs/{}/'.format( exp_id )
    if not os.path.exists(path):
        os.makedirs(path)
        np.savez(path+'meta_data.npz', exp_id=exp_id, uext_array=uext_array, ie_ratio_array=ie_ratio_array)
    
    file_name = str(indx).zfill(3) +'_'+str(int(time.time()))
    
    np.savez(path +'{}.npz'.format(file_name), config=config, mnn_mean=u,mnn_std=s)


    #with open(path +'{}_config.json'.format(file_name),'w') as f:
    #    json.dump(config,f)

    #runfile('./batch_processor.py', args = '0 test', wdir='./')
