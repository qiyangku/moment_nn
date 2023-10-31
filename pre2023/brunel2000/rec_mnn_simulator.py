# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 13:45:19 2020

@author: dell
"""

import matplotlib.pyplot as plt
from mnn_core.maf import MomentActivation
import numpy as np
#import cupyx as cp
import time


#np.random.seed(1)
class InputGenerator():
    def __init__(self, config):
        self.NE = config['NE']
        self.NI = config['NI']
        self.N = config['NE']+config['NI']
        self.uext = config['uext']
        we = config['wee']['mean']
        #define external input mean
        #
        self.input_mean = we*self.uext*np.ones((self.N,1)) #input current mean
        
        #calculate external input cov (assume independent Poisson spikes)
        self.input_cov = we*we*self.uext*np.eye(self.N) 
        
        #self.L_ext = np.linalg.cholesky(self.input_cov)        
        
        return

class RecurrentMNN():
    def __init__(self, config, W, input_gen):
        self.NE = config['NE']
        self.NI = config['NI'] # number of neuron per layer
        self.N = self.NE + self.NI
        self.dt = 0.1 #integration time-step
        self.tau = 1 #synaptic? time constant
        
        self.W = W #synaptic weight matrix (csr)
        self.Wt = W.T # store transpose (csc)
        self.input_gen = input_gen # input generator, class object
        self.maf = MomentActivation()
        self.maf.Vth = config['Vth']
        self.maf.Vres = config['Vres']
        self.maf.Tref = config['Tref']
    
    
    def summation(self, mean_in, std_in, corr_in):
        '''INPUTS: input mean/std/corr, stim = stimulus value'''
        
        
        
        mean_out = self.W.dot(mean_in)
        
        mean_out += self.input_gen.input_mean #external input mean
        
        C_in = std_in.reshape(1,self.N)*corr_in*std_in.reshape(self.N,1) #covariance matrix
        
        
        '''WARNING: inconsist dot product expression between np and sp')
        see https://stackoverflow.com/questions/48243928/matrix-multiplication-of-a-sparse-scipy-matrix-with-two-numpy-vectors
        '''
        #C_out = self.W.dot(C_in)
        #C_out = C_out*self.Wt  # stupid scipy uses * for matrix multiplication!
        
        C_out = self.W @ C_in @ self.Wt #requires python 3.5+, tested it's correct answer
        
        C_out += self.input_gen.input_cov
        
        std_out = np.sqrt(np.diag(C_out).copy().reshape(self.N,1))
        
        #std_out = np.maximum(1e-16,std_out)
        corr_out = C_out/std_out.reshape(self.N,1)/std_out.reshape(1,self.N)
        
        np.fill_diagonal(corr_out,1.0)
        #for i in range(self.N):
        #    corr_out[i,i] = 1.0
            
        corr_out[np.isnan(corr_out)] = 0.0
        
        return mean_out, std_out, corr_out
        
    

    
    def forward(self, mean_in, std_in, corr_in):
        mean_out, std_out, corr_out = self.summation(mean_in, std_in, corr_in)
               
        u = self.maf.mean( mean_out, std_out)
        s,_ = self.maf.std( mean_out, std_out)
        chi = self.maf.chi( mean_out, std_out)
        rho = chi.reshape(self.N,1)*chi.reshape(1,self.N)*corr_out
        np.fill_diagonal(rho,1.0)
        #for i in range(self.N):
        #    rho[i,i]=1.0
        return u, s, rho
    
    
    def run(self, T = 10, record_ts = True):
        self.nsteps = int(T/self.dt)
        
        # initial condition
        u = np.zeros((self.N,1))
        s = np.zeros((self.N,1))
        rho = np.eye(self.N)
        
        if record_ts: # record time series data; turn off if out of memory
            U = np.zeros((self.N, self.nsteps))
            S = np.zeros((self.N, self.nsteps))
            R = np.zeros((self.N, self.N, self.nsteps))
            
        a = self.dt/self.tau        
        
        
        for i in range(self.nsteps):
            #print('Starting iteration {}/{}'.format(i,self.nsteps))
            #print('i={}, s.shape = {}'.format(i,s.shape)  )
            if record_ts:
                U[:,i] = u.ravel()
                S[:,i] = s.ravel()
                R[:,:,i] = rho
            
            maf_u, maf_s, maf_rho = self.forward(u, s, rho)                      
            
            C = rho*np.reshape(s,(1,self.N))*np.reshape(s,(self.N,1))
            maf_C = maf_rho*np.reshape(maf_s,(1,self.N))*np.reshape(maf_s,(self.N,1))
            
            u = (1-a)*u + a*maf_u
            C = (1-a)*C + a*maf_C
            
            s = np.sqrt(np.diagonal(C).copy())
            rho = C/s.reshape(1,self.N)/s.reshape(self.N,1)
            
        if record_ts:
            return U, S, R
        else:
            return u, s, rho
    

        

    # def plot_grid(self, dat):
    #     size = dat[2].shape
    #     fig, axes = plt.subplots(size[0], size[1])
    #     fig2, axes2 = plt.subplots(size[0], size[1])
    #     for i in range(size[0]):
    #         for j in range(size[1]):
    #             u = dat[2][i,j,:]
    #             s = dat[3][i,j,:]
    #             rho = dat[4][i,j,:,:]
                
    #             axes[i,j].plot(self.x, u)
    #             axes[i,j].plot(self.x, s*s)
    #             axes[i,j].set_ylim(0,0.1)
    #             #axes[i,j].set_xticks([])
                
    #             mean_current, std_current, corr_current = self.summation(u,s,rho)
    #             axes2[i,j].plot(self.x, mean_current)
    #             axes2[i,j].plot(self.x, std_current)
                
    #             #axes[i,j].set_title('we={}, IE_ratio={}'.format(dat[0][i],dat[0][j]))
                
        
    