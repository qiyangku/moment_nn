# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 21:29:22 2023

@author: Yang Qi

Threshold-integration scheme for solving Fokker-Planck equation and flux.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from pre2023.utils import perf_timer

class FPE_solver():    
    def __init__(self):
        '''
        Solve stationary FPE associated with current-based leaky LIF neuron model.
    
        '''
        
        # neuron model parameter
        
        self.Vth = 20
        self.Vref = 0            
        self.tau = 20 # membrane time constant 20 ms
        self.El = 0 # leak reversal potential
        self.Tref = 5 # ms
        
        # FPE solver parameter
        self.Vlb = -20 # lower bound for solving FPE, not model parameter
        self.nV = 1000
        
    
    @perf_timer
    def run(self, u0, var):
        '''    
        INPUT: u0, var = mean/var of total synaptic current
        OUTPUT: r = mean firing rate
        '''
        
        V = np.linspace(self.Vlb, self.Vth, self.nV)
        P = np.zeros( (len(u0),  self.nV) )
        
        dV = (self.Vth - self.Vlb)/(self.nV-1)        
        kref = int((self.Vref - self.Vlb)/dV) # index of refractory potential;
        
        pp = 0
        j = 1
        
        for k in reversed(range(self.nV)):
            v = V[k]
            P[:,k] = pp
            
            u = (v - self.El)/self.tau - u0
            
            #>>integration
            G=dV*u/var*2;
            A = np.exp(G);
            B=(A-1)/G;
            B[G==0]=1;
            B=dV*B/var*2;
            
            pp = pp*A + j*B;
            j = j - (k==(kref+1));
            
        # firing rate
        r = 1/(self.Tref + np.sum(P, axis = 1)*dV)
        
        return r
        

if __name__=='__main__':
    solver = FPE_solver()
    
    m = 100
    u0 = np.linspace(-0.5,2.5,m)
    var = np.ones(m)
    
    r1, dt1 = solver.run(u0,var)
    print(dt1)
    r2, dt2 = solver.run(u0,var*4)
    print(dt2)
    
    plt.close('all')
    plt.plot(u0,r1,u0,r2)
    
    # good. result is identical to MA
    
    
