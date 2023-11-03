# -*- coding: utf-8 -*-
"""
For testing the MA with non-stationary inputs

@author: dell
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time
from mnn_core.maf import *

class InteNFire():
    def __init__(self, num_neurons = 10):
        self.L = 1/20 #ms
        self.Vth = 20
        self.Vres = 0        
        self.Tref = 5 #ms
        self.Vspk = 50
        self.dt = 1e-3 #integration time step (ms)
        self.num_neurons = num_neurons
        #self.T = T    
        self.we = 1 #synaptic weight; set to 1 so that w = w^2;  
        self.maf = MomentActivation()
        
        # inhomogeneous input parameters
        self.u_ext = 1 # kHz average rate        
        self.phase = 2*np.pi*np.random.rand(num_neurons) # random phase shift        
        self.A = 1 # same as u_ext by default, A must be smaller than u_ext
        #self.T = 10e3 #ms 
    
    def input_spikes(self, t):
                
        r = self.u_ext*(1 + self.A*np.sin( 2*np.pi*self.freq*t + self.phase) )
        is_spk = np.random.rand(self.num_neurons) < r*self.dt
        
        return is_spk
        
        
    
    def run(self, T = 1e3, freq = 0.01, ntrials = None, record_v = False, show_message = False):
        '''Simulate integrate and fire neurons'''
        
        self.T=T
        self.freq = freq
        
        num_timesteps = int(self.T/self.dt)
        
        tref = np.zeros(self.num_neurons) #tracker for refractory period
        v = np.random.rand(self.num_neurons)*self.Vth #initial voltage
        #v = np.zeros(self.num_neurons)
        
        SpkTime = [[] for i in range(self.num_neurons)]
        t = np.arange(0, self.T , self.dt)
        if record_v:
            V = np.zeros( (self.num_neurons, num_timesteps) )
        else:            
            V = None
                
        start_time = time.time()
        for i in range(num_timesteps):

            input_current = self.we*self.input_spikes(i*self.dt)                 
         
            v += -v*self.L*self.dt + input_current
            
            #check state
            is_ref = (tref > 0.0) & (tref < self.Tref)
            is_spike = (v > self.Vth) & ~is_ref
            is_sub = ~(is_ref | is_spike) 
            
            v[is_spike] = self.Vspk            
            v[is_ref] = self.Vres
            
            #update refractory period timer
            tref[is_sub] = 0.0
            tref[is_ref | is_spike] += self.dt
            
            
            if record_v:
                V[:,i] = v            
            
            for k in range(self.num_neurons):
                if is_spike[k]:
                    SpkTime[k].append(i*self.dt)
            
            if show_message and (i+1) % int(num_timesteps/10) == 0:
                progress = (i+1)/num_timesteps*100
                elapsed_time = (time.time()-start_time)/60
                print('Progress: {:.2f}%; Time elapsed: {:.2f} min'.format(progress, elapsed_time ))
                
        return SpkTime, V, t
    
    def empirical_maf(self, SpkTime):
        '''Turns out it's a bad idea to calulate spk stats with isi'''
        # isi = []
        # for spk_time in SpkTime:
        #     spk_time = np.array(spk_time)
        #     spk_time = spk_time[ spk_time > 500] #remove burn-in time; unit: ms
        #     if spk_time.size > 1:
        #         isi.extend( list(np.diff(spk_time)) )
            
        # if len(isi)>30:
        #     mean_isi = np.mean(isi)
        #     var_isi = np.var(isi)
        #     mu = 1/mean_isi        
        #     sig = np.sqrt( np.power(mu,3)*var_isi )
        # else:
        spike_count = [len(spk_time) for spk_time in SpkTime]            
        mu = np.mean(spike_count)/self.T
        sig = np.sqrt(np.var(spike_count)/self.T)
        
        
        return mu, sig

def parameter_sweep(save_results = False):
    ''' SNN simulation with different input oscillation frequencies'''    
    num_trials = 1000
    T = 1e3 # ms    
    # range of frequencies? Biological range: delta (1-4 Hz), theta (4-8 Hz), beta (13-30 Hz), gamma (30-150 Hz)
    # take T=1e3 as long window, T=1e2 => 1e-3 KHz
    
    freqs = 1e-3*np.logspace(-1,2,100) #unit: kHz (23.37 min for 10 freqs; I should 4x this)
    
    inf = InteNFire(num_neurons = num_trials) #time unit: ms        
    
    SpkTimes = []
    
    start_time = time.time()
    for i in range(freqs.size):
        spktime, _, _ = inf.run(T = T, freq = freqs[i], show_message = False, record_v=False)
        SpkTimes.append(spktime)
                
        progress = (i +1)/freqs.size*100
        elapsed_time = (time.time()-start_time)/60
        print('Progress: {:.2f}%; Time elapsed: {:.2f} min'.format(progress, elapsed_time ))
    
    SpkTimes = np.array(SpkTimes, dtype=object)
    
    if save_results:
        np.savez('./runs/snn_inhom_input.npz', SpkTimes=SpkTimes, freqs=freqs, num_trials=num_trials, T=T)
    return SpkTimes, freqs, num_trials, T

def parameter_sweep_amplitude(save_results = False):
    ''' SNN simulation with different input oscillation amplitudes (but with fixed frequency)'''    
    num_trials = 1000
    T = 1e3 # ms    
    # range of frequencies? Biological range: delta (1-4 Hz), theta (4-8 Hz), beta (13-30 Hz), gamma (30-150 Hz)
    # take T=1e3 as long window, T=1e2 => 1e-3 KHz
    
    amps = np.linspace(0,1,100) #!!! debug # samplitude of input oscillation # when c=0, the input is stationary and MA should work perfectly
    freq = 0.0014174741629268055 # resonance frequency (see snn_inhom_input.npz)
    inf = InteNFire(num_neurons = num_trials) #time unit: ms
    
    SpkTimes = []
    
    start_time = time.time()
    for i in range(amps.size):
        inf.A = amps[i]
        spktime, _, _ = inf.run(T = T, freq = freq, show_message = False, record_v=False)
        SpkTimes.append(spktime)
                
        progress = (i +1)/amps.size*100
        elapsed_time = (time.time()-start_time)/60
        print('Progress: {:.2f}%; Time elapsed: {:.2f} min'.format(progress, elapsed_time ))
    
    SpkTimes = np.array(SpkTimes, dtype=object)
    
    if save_results:
        np.savez('./runs/snn_inhom_input_vary_amp.npz', SpkTimes=SpkTimes, amps=amps, freq=freq, num_trials=num_trials, T=T)
    return SpkTimes, amps, num_trials, T


def spk_data_analysis():
    dat = np.load('./runs/snn_inhom_input.npz', allow_pickle=True)
    SpkTimes = dat['SpkTimes']
    freqs = dat['freqs']
    num_trials = dat['num_trials']
    T = dat['T']
    
    nbins = 100 # increment 10 ms
    mu = np.zeros((len(freqs), nbins))
    var = np.zeros((len(freqs), nbins))
    bin_edges = np.linspace(0,T, nbins+1)
    for i in range(freqs.size):
        spkcount = np.zeros((num_trials, nbins))
        for j in range(len(SpkTimes[i])):         
            h, _ = np.histogram(SpkTimes[i][j], bin_edges )
            spkcount[j,:] = np.cumsum(h)
        mu[i,:] = np.mean(spkcount, axis=0)/bin_edges[1:]
        var[i,:] = np.var(spkcount, axis=0)/bin_edges[1:]
    
    return mu, var, freqs, bin_edges[1:]

def theoretical_prediction(dT, u_ext, w, c):
    ''' dT = spike count time windows, 
    u_ext = avg Poisson rate
    w = str of synaptic weight
    c = str of inhomogeneity
    '''
    maf = MomentActivation()
    
    u_in = w*u_ext*np.ones( dT.size )
    s_in_short = w*np.sqrt(u_ext + 0.5*u_ext*u_ext*c*c*dT)
    s_in_long = w*np.sqrt(u_ext)*np.ones( dT.size )
    
    u_out_short_T = maf.mean(u_in, s_in_short)
    s_out_short_T, _ = maf.std(u_in, s_in_short)
    
    u_out_long_T = maf.mean(u_in, s_in_long)    
    s_out_long_T, _ = maf.std(u_in, s_in_long)
    
    return u_out_short_T, s_out_short_T, u_out_long_T, s_out_long_T
    
    

if __name__=='__main__':

    # # SIMULATION
    #parameter_sweep(save_results=True)
    parameter_sweep_amplitude(save_results=True)
    # ANALYSIS
    #mu, var, freqs, readout_time = spk_data_analysis()    
    #u_out_short_T, s_out_short_T, u_out_long_T, s_out_long_T = theoretical_prediction(readout_time, 1, 1, 1)
    
    # # #################
    # # # VISUALIZATION
    # dT_indx = 99 # 10 in total
    # print('Readout time (ms): ', readout_time[dT_indx])
    # plt.close('all')
    # plt.figure(figsize=(7,3))
    # plt.subplot(1,2,1)
    # plt.semilogx(freqs, mu[:,dT_indx]) # empirical result
    # plt.semilogx(  [freqs[0] , freqs[-1]], u_out_long_T[dT_indx]*np.ones(2)   ,'--') # long readout time, fast oscillation
    # #plt.semilogx(  [freqs[0] , freqs[-1]], u_out_short_T[dT_indx]*np.ones(2)  ,'--' ) # short readout time, slow oscillation
    # plt.ylim([0, 0.025])
    # plt.ylabel('Mean firing rate (kHz)')
    # plt.xlabel('Oscillation frequency (kHz)')
    
    # plt.subplot(1,2,2)
    # plt.semilogx(freqs, var[:,dT_indx]/mu[:,dT_indx]) # dT =1000 ms
    # plt.semilogx(  [freqs[0] , freqs[-1]], s_out_long_T[dT_indx]**2/u_out_long_T[dT_indx]*np.ones(2)  ,'--' )
    # #plt.semilogx(  [freqs[0] , freqs[-1]], s_out_short_T[dT_indx]**2*np.ones(2)  ,'--' )
    # #plt.ylim([0, 0.025])
    # plt.xlabel('Oscillation frequency (kHz)')
    # plt.ylabel('Fano factor')
    # plt.ylim([])
    # #plt.semilogx(freqs, var[:,-1])
    
    # plt.tight_layout()
    
    # # ################
    
    
    # plt.close('all')
    # plt.figure()
    # plt.plot(readout_time, u_out_short_T)
    # plt.plot(readout_time, s_out_short_T**2)
    # plt.xlabel('Readout time (ms)')
    # plt.title('Short time window prediction')
    # #plt.ylabel('Mean firing rate (sp/ms)')
    
    # plt.figure()
    # plt.plot(readout_time, u_out_long_T)
    # plt.plot(readout_time, s_out_long_T**2)    
    # plt.xlabel('Readout time (ms)')
    # plt.title('Long time window prediction')
    # #plt.ylabel('Firing variability')
    
    print('\007') #make a sound when finish
        