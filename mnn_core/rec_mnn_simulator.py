# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 13:45:19 2020

@author: dell
"""

import matplotlib.pyplot as plt
from mnn_core.maf import MomentActivation
import numpy as np
import time


#np.random.seed(1)

class RecurrentMNN():
    def __init__(self, config, W, input_gen):
        self.NE = config['NE']
        self.NI = config['NI'] # number of neuron per layer
        self.N = self.NE + self.NI
        self.dt = 0.1 #integration time-step
        self.tau = 1 #synaptic? time constant
        
        # if isinstance(config['u_ext'], (int,float)):
        #     self.u_ext = np.zeros((self.N,1))
        #     self.u_ext[:] = config['u_ext'] # only E neurons receive input
        # else:
        #     self.u_ext = config['u_ext']
        #  #external current
        
        # if isinstance(config['s_ext'], (int,float)):
        #     self.s_ext = np.zeros((self.N,1))
        #     self.s_ext[:] = config['s_ext']
        # else:
        #     self.s_ext = config['s_ext']
            
        # if config['corr_ext'] is not None:
        #     self.corr_ext = config['corr_ext'] #external correlation coefficient
        # else:
        #     self.corr_ext = np.eye( self.N ) 
        
        self.W = W # synaptic weight matrix (csr)
        self.Wt = W.T # store transpose (csc)
        self.input_gen = input_gen # input generator, class object
        self.maf = MomentActivation()
        self.maf.Vth = config['Vth']
        
        
    # def synaptic_weight(self, config):
    #     N = self.N
    #     "Homogeneous population"
    #     if config['randseed'] is not None:
    #         rng = np.random.default_rng( config['randseed'] )
    #         W = rng.standard_normal(size=(N, N))
    #         coin_toss = rng.uniform(size=(N,N))
    #     else:
    #         W = np.random.randn(N,N)
    #         coin_toss = np.random.rand(N,N)
        
    #     #apply synaptic weight
    #     #   excitatory weight
    #     W[:,:self.NE] = W[:,:self.NE]*config['we']['std'] + config['we']['mean']
    #     W[:,:self.NE] = np.abs(W[:,:self.NE])
    #     #   inhibitory weight
    #     W[:,self.NE:] = W[:,self.NE:]*config['wi']['std'] + config['wi']['mean']
    #     W[:,self.NE:] = -np.abs(W[:,self.NE:])
        
    #     #apply connection probability (indegree should then be poisson)
    #     W[ coin_toss > config['conn_prob'] ] = 0
        
    #     return W
    
    def summation_no_corr(self, mean_in, std_in, stim):
        mean_out = self.W.dot(mean_in)
        mean_out += self.input_gen.input_mean(stim) #external input mean
        
        var_out = (self.W**2) @ (std_in**2) + self.input_gen.var_ind + self.input_gen.var_shr
        
        return mean_out, np.sqrt(var_out)
    
    def summation(self, mean_in, std_in, corr_in, stim):
        '''INPUTS: input mean/std/corr, stim = stimulus value'''
        
        
        
        mean_out = self.W.dot(mean_in)
        
        mean_out += self.input_gen.input_mean(stim) #external input mean
        
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
        
    
    def forward_no_corr(self, mean_in, std_in, stim):
        mean_out, std_out = self.summation_no_corr(mean_in, std_in, stim)
        u = self.maf.mean( mean_out, std_out)
        s,_ = self.maf.std( mean_out, std_out)
        return u, s
    
    def forward(self, mean_in, std_in, corr_in, stim):
        mean_out, std_out, corr_out = self.summation(mean_in, std_in, corr_in, stim)
               
        u = self.maf.mean( mean_out, std_out)
        s,_ = self.maf.std( mean_out, std_out)
        chi = self.maf.chi( mean_out, std_out)
        rho = chi.reshape(self.N,1)*chi.reshape(1,self.N)*corr_out
        np.fill_diagonal(rho,1.0)
        #for i in range(self.N):
        #    rho[i,i]=1.0
        return u, s, rho
    
    def run_no_corr(self, T=10, stim = 0.1825, record_ts = True):
        self.nsteps = int(T/self.dt)
        
        # initial condition
        u = np.zeros((self.N,1))
        s = np.zeros((self.N,1))
        
        if record_ts: # record time series data; turn off if out of memory
            U = np.zeros((self.N, self.nsteps))
            S = np.zeros((self.N, self.nsteps))
            
        a = self.dt/self.tau
        
        for i in range(self.nsteps):
            if record_ts:
                U[:,i] = u.ravel()
                S[:,i] = s.ravel()
            
            maf_u, maf_s = self.forward_no_corr(u, s, stim)                      
            
            u = (1-a)*u + a*maf_u
            s = (1-a)*s + a*maf_s
            
        if record_ts:
            return U, S
        else:
            return u, s
    
    def run(self, T = 10, stim = 0.1825, record_ts = True):
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
            #print('i={}, s.shape = {}'.format(i,s.shape)  )
            if record_ts:
                U[:,i] = u.ravel()
                S[:,i] = s.ravel()
                R[:,:,i] = rho
            
            maf_u, maf_s, maf_rho = self.forward(u, s, rho, stim)                      
            
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
    
    def para_sweep(self):
        '''Do a parameter sweep over the weight space'''
        WE = np.linspace(0.0 , 10.0 ,11)
        ie_ratio = np.linspace(0.0 , 1.0 ,10)
        
        U = np.zeros( (len(WE), len(ie_ratio), self.N) )
        S = np.zeros( (len(WE), len(ie_ratio), self.N) )
        R = np.zeros( (len(WE), len(ie_ratio), self.N, self.N) )
        
        T = 10
        t0 = time.perf_counter()
        for i in range(len(WE)):
            for j in range(len(ie_ratio)):
                self.W, self.w = self.mexi_mat(h = WE[i], ie_ratio = ie_ratio[j])     
                u,s,r = self.run(T)
                U[i,j,:] = u[:,-1]
                S[i,j,:] = s[:,-1]
                R[i,j,:,:] = r[:,:,-1]
            print('WE={}, ie_ratio={}'.format(WE[i],ie_ratio[j]))
            print('Time Elapsed: ',time.perf_counter() -t0 )
        return WE, ie_ratio, U, S, R
        

    def plot_grid(self, dat):
        size = dat[2].shape
        fig, axes = plt.subplots(size[0], size[1])
        fig2, axes2 = plt.subplots(size[0], size[1])
        for i in range(size[0]):
            for j in range(size[1]):
                u = dat[2][i,j,:]
                s = dat[3][i,j,:]
                rho = dat[4][i,j,:,:]
                
                axes[i,j].plot(self.x, u)
                axes[i,j].plot(self.x, s*s)
                axes[i,j].set_ylim(0,0.1)
                #axes[i,j].set_xticks([])
                
                mean_current, std_current, corr_current = self.summation(u,s,rho)
                axes2[i,j].plot(self.x, mean_current)
                axes2[i,j].plot(self.x, std_current)
                
                #axes[i,j].set_title('we={}, IE_ratio={}'.format(dat[0][i],dat[0][j]))
                
#%% 
if __name__ == "__main__":
    config = {
    'NE': 100,
    'NI': 100,
    'dt': 0.1,
    'tau': 1,
    'u_ext': 1.0,
    's_ext': 1.0,
    'corr_ext': None,
    'we':{'mean': 0.1, 'std': 0.05},
    'wi':{'mean': 0.4, 'std': 0.2},
    'conn_prob': 0.3, #connection probability
    'randseed': 0
    }
    
    
    sfc = HomogeneousRNN(config) 
    U,S,R=sfc.run(20)
    
    #plt.clf()
    plt.figure(figsize=(5,8))
    
    plt.subplot(4,1,1)
    plt.imshow(U)
    plt.subplot(4,1,2)
    plt.imshow(S)
    
    x = np.arange(sfc.N)
    
    
    plt.subplot(4,1,3)
    plt.plot( x, U[:,-1], x, S[:,-1]**2)
    #plt.ylim(0,0.1)
    plt.xlabel('neuron index')
    plt.ylabel('neural activity')
    plt.legend(('mean','var'))
    
    plt.subplot(4,1,4)
    mean_current, std_current, corr_current = sfc.summation(U[:,-1],S[:,-1],R[:,:,-1])
    plt.plot( x, mean_current, x, std_current  )
    plt.xlabel('neuron index')
    plt.ylabel('total synaptic current')
    plt.legend(('mean','var'))
    
    plt.tight_layout()

    
    
#%%    
    m = dat[4].shape[0]
    n = dat[4].shape[1]
    num_neurons = dat[4].shape[3]
    
    rho_max = np.zeros((m,n))
    std_max = rho_max.copy()
    
    mean_max = np.zeros((m,n))
    
    for i in range(m):
        for j in range(n):
            rho = dat[4][i,j,:,:] - np.eye(num_neurons)
            rho_max[i,j] = np.max(np.abs(rho))
            std_max[i,j] = np.max(dat[3][i,j,:])            
            mean_max[i,j] = np.max(dat[2][i,j,:])
            
            
    fig = plt.figure()        
    
    ax1 = fig.add_subplot(311)
    pos = ax1.imshow(mean_max, aspect='auto', origin='lower', extent = [dat[0][0], dat[0][-1], dat[1][0],dat[1][-1]])
    ax1.set_ylabel('Excitatory weight')
    fig.colorbar(pos, ax=ax1)
    ax1.set_title('Max mean')
    
    ax3 = fig.add_subplot(312)
    pos3 = ax3.imshow(std_max, aspect='auto', origin='lower', extent = [dat[0][0], dat[0][-1], dat[1][0],dat[1][-1]])
    ax3.set_ylabel('Excitatory weight')
    fig.colorbar(pos3, ax=ax3)
    ax3.set_title('Max std')
    
    
    ax2 = fig.add_subplot(313)
    pos2 = ax2.imshow(rho_max, aspect='auto', origin='lower', extent = [dat[0][0], dat[0][-1], dat[1][0],dat[1][-1]])
    fig.colorbar(pos2, ax=ax2)
    ax2.set_ylabel('Excitatory weight')
    ax2.set_xlabel('I-E weight ratio')
    ax2.set_title('Max corr. coef. (off-diagonal)')
    
    #======================='
    
    
    # U, S, R = sfc.run()
    
    # fig = plt.figure()        

    # ax1 = fig.add_subplot(211)
    # ax1.imshow(U,extent = [0,sfc.M,0,sfc.N],origin='lower')#,interpolation="lanczos")#
    # ax1.set_xlabel('Layers')
    # ax1.set_ylabel('Neurons')
    # ax1.set_title('Mean')            
    # ax1.set_aspect('auto')
    
    # ax2 = fig.add_subplot(212)
    # ax2.imshow(S,extent = [0,sfc.M,0,sfc.N],origin='lower')#,interpolation="lanczos")#extent = [ubar[0],ubar[-1],sbar[0],sbar[-1]],
    # ax2.set_xlabel('Layers')
    # ax2.set_ylabel('Neuron')   
    # ax2.set_title('Std')
    # ax2.set_aspect('auto')
    
    # fig.subplots_adjust(hspace = 0.6)
    
    # fig2 = plt.figure()
    
    # for i in range(9):
    #     ax = fig2.add_subplot('33{}'.format(i+1))
    #     #layer_num = int(i*10)
    #     layer_num = i
    #     ax.imshow(R[:,:,layer_num],vmin=-1,vmax=1)# - np.eye(sfc.N))#,interpolation="lanczos")#extent = [ubar[0],ubar[-1],sbar[0],sbar[-1]],
    #     ax.axis('off')
    #     #ax2.set_xlabel('Layers')
    #     #ax2.set_ylabel('Neurons')   
    #     ax.set_title('Layer {}'.format(layer_num))
   
        
    