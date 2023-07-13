# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 16:11:47 2021

@author: dell
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time



class InteNFireRNN():
    def __init__(self, config, W, input_gen):
        ''' W = weight matrix, input_gen = generator for external input'''
        self.L = 1/20 #ms
        self.Vth = config['Vth']
        self.Vres = 0        
        self.Tref = config['Tref'] #ms
        self.Vspk = 50 #for visualization purposes only
        self.dt = 1e-1 #integration time step (ms)
        self.NE = config['NE']
        self.NI = config['NI']
        
        self.num_neurons = config['NE'] + config['NI']      
        
        # if isinstance(config['u_ext'], (int,float)):
        #     self.u_ext = np.zeros((self.num_neurons,1))
        #     self.u_ext[:] = config['u_ext'] # only E neurons receive input
        # else:
        #     self.u_ext = config['u_ext']
        #  #external current
        
        # if isinstance(config['s_ext'], (int,float)):
        #     self.s_ext = np.zeros((self.num_neurons,1))
        #     self.s_ext[:] = config['s_ext']
        # else:
        #     self.s_ext = config['s_ext']
            
        # self.corr_ext = config['corr_ext'] #external correlation coefficient
        
        
        self.W = W #synaptic weight matrix
        self.input_gen = input_gen # input generator, class object
    
    # def input_spk_time(self, spk_mean, spk_var):
    #     '''
    #         generate gamma distributed spike time
    #         isi_mean = shape*scale (unit: kHz)
    #         isi_var = shape*scale^2 (unit: kHz)
    #     '''
        
    #     isi_mean = 1/spk_mean
    #     isi_var = np.power(isi_mean,3)*spk_var
        
    #     scale = isi_var/isi_mean
    #     shape = isi_mean*isi_mean/isi_var
        
    #     num_spikes = int(self.T/isi_mean)*5
    #     num_samples = num_spikes*self.num_neurons
        
    #     isi = np.random.gamma(shape, scale, num_samples)
    #     isi = isi.reshape((self.num_neurons, num_spikes ))
        
    #     spk_time = np.cumsum(isi, axis=1)
        
    #     return spk_time         
    
    # def input_ei_current(self, t):
    #     '''Convert intput spike (sparse matrix) to post-synaptic current'''
    #     current = self.we*self.exc_input[:,t] - self.wi*self.inh_input[:,t]        
    #     return current.toarray().flatten()
        
    
    # def input_spike_sparse(self, spk_mean, spk_var):
        
    #     scale = spk_var/spk_mean/spk_mean
    #     shape = spk_mean/spk_var
        
    #     num_spikes = int(self.T*spk_mean)*5
    #     num_samples = num_spikes*self.num_neurons
        
    #     isi = np.random.gamma(shape, scale, num_samples)
    #     isi = isi.reshape((self.num_neurons, num_spikes ))        
        
    #     spk_time = np.cumsum(isi, axis=1)
        
    #     #need to do a safety check to make sure that spk_time[:,-1] > self.T for all neurons
    #     if np.sum(spk_time[:,-1] < self.T):
    #         print('Warning: not enough spikes!')
        
    #     spk_time = np.floor( spk_time/self.dt).flatten()
    #     neuron_index = np.tile( np.arange(self.num_neurons) , (num_spikes,1)).T.flatten()        
    #     dat = np.ones(num_samples)
        
    #     spk_mat = sp.sparse.coo_matrix( (dat, (neuron_index, spk_time)), shape = (self.num_neurons, int(np.max(spk_time))+1 ) ).tocsc() #compressed column format
        
        
    #     spk_mat = spk_mat[:,:int(self.T/self.dt)] 
    #     return spk_mat
        
    
    # def input_gaussian_current(self, mean, std, corr = None):
    #     ''' Generate gaussian input current '''
    #     N = len(mean)
    #     if corr is None:
    #         input_current = np.random.randn(self.num_neurons).reshape(N,1)
    #         input_current = input_current*std*np.sqrt(self.dt) + mean*self.dt
    #         #input_current = input_current
    #     else:
    #         cov = corr*std.reshape(N,1)*std.reshape(1,N)
    #         input_current = np.random.multivariate_normal(mean.ravel()*self.dt, cov*self.dt)   
    #     return input_current.reshape(N,1)
    
    
    def forward(self, v, tref, is_spike, ff_current):
        #compute voltage
        v += -v*self.L*self.dt + (self.W @ is_spike) + ff_current
        
        #compute spikes
        is_ref = (tref > 0.0) & (tref < self.Tref)
        is_spike = (v > self.Vth) & ~is_ref
        is_sub = ~(is_ref | is_spike)
                
        v[is_spike] = self.Vspk            
        v[is_ref] = self.Vres
        
        #update refractory period timer
        tref[is_sub] = 0.0
        tref[is_ref | is_spike] += self.dt
        return v, tref, is_spike
        
    
    
    def run(self, T, s = None, ntrials = None, record_v = False, show_message = False):
        '''Simulate integrate and fire neurons
        T = simulation duration
        s = stimulus value        
        '''
        
        
        
        self.T = T#min(10e3, 100/maf_u) #T = desired number of spikes / mean firing rate
        num_timesteps = int(self.T/self.dt)
        
        tref = np.zeros((self.num_neurons,1)) #tracker for refractory period
        #v = np.random.rand(self.num_neurons,1)*self.Vth #initial voltage
        v = np.zeros( (self.num_neurons,1) )
        is_spike = np.zeros(v.shape)
        
        SpkTime = [[] for i in range(self.num_neurons)]
        t = np.arange(0, self.T , self.dt)
        
        if record_v:
            V = np.zeros( (self.num_neurons, num_timesteps) )
        else:            
            V = None
                
        # mean, std, rho = input_mean, input_std, input_corr # input current stats (not firing rate, it is current)
        
        # if input_type == 'spike':
        #     #fix mean_inh and std_inh
        #     inh_var = self.ei_balance*np.power(std/self.wi,2)
        #     exc_var = (1-self.ei_balance)*np.power(std/self.we,2)
            
        #     exc_mean = (mean + self.inh_curr_mean)/self.we
        #     inh_mean = self.inh_curr_mean/self.wi
            
        #     self.exc_input = self.input_spike_sparse(exc_mean, exc_var)
        #     self.inh_input = self.input_spike_sparse(inh_mean, inh_var)
        
        start_time = time.time()
        for i in range(num_timesteps):
            
            input_current = self.input_gen.gen_gaussian_current(s, self.dt)
            #evolve forward in time
            #if input_type == 'gaussian':
            #    input_current = self.input_gaussian_current( self.u_ext, self.s_ext, self.corr_ext)
            # elif input_type == 'bivariate_gaussian':
            #     corr = np.eye(2) + rho*(1-np.eye(2))
            #     cov = corr*std[:2].reshape(2,1)*std[:2].reshape(1,2)
            #     input_current = np.random.multivariate_normal(mean[:2]*self.dt, cov*self.dt, size = ntrials) #output is ntrials-by-2
            #     input_current = input_current.flatten()
            #     #corr = np.kron( np.eye( int(mean.size/2) ), corr) #create block diagonal matrix
            #     #input_current = self.input_gaussian_current(mean, std, corr = corr)
            #     #don't do this too slow.
            #     # generate bivariate gaussian of shape (2, num of samples) then reshape it
                
                
            # elif input_type == 'multivariate_gaussian':
            #     corr = np.eye(self.num_neurons) + rho*(1-np.eye(self.num_neurons))
            #     input_current = self.input_gaussian_current(mean, std, corr = corr)
            # elif input_type == 'spike':
            #     #input_current = self.input_synaptic_current(i*self.dt, spk_time)
            #     input_current = self.input_ei_current(i)                   
                
            
            v, tref, is_spike = self.forward(v, tref, is_spike, input_current)
            
            if record_v:
                V[:,i] = v            
            
            for k in range(self.num_neurons):
                if is_spike[k]:
                    SpkTime[k].append(i*self.dt)
            
            if show_message and (i+1) % int(num_timesteps/10) == 0:
                progress = (i+1)/num_timesteps*100
                elapsed_time = (time.time()-start_time)/60
                print('Progress: {:.2f}%; Time elapsed: {:.2f} min'.format(progress, elapsed_time ), flush=True)
                
        return SpkTime, V, t

def spk_time2count(SpkTime, T, binsize = 100):
    '''Calculate Fano factor from spike time'''
    nbins = int( T/binsize)
    num_neurons = len(SpkTime)
    spk_count = np.zeros((num_neurons, nbins))
    #print(spk_count.shape)
    for i in range(num_neurons):
        h, bin_edges = np.histogram( SpkTime[i], nbins , range=(0,T))
        spk_count[i,:] = h
    return spk_count
    
    
