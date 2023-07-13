# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 21:29:22 2023

@author: Yang Qi

Generate or load direct look-up table with interpolation
"""
from mnn_core.maf import MomentActivation
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import matplotlib.pyplot as plt
import time
from pre2023.utils import perf_timer

# direct quadrature
# use ma.ds1.brute_force


class LookUpSolver():    
    def __init__(self):
        '''
        Generate or load direct look-up table with interpolation
    
        '''
        self.file = 'ma_lookup_tab.npz'
        self.num_pts = 1000 #number of points (along one dimension)
        self.input_mean = np.linspace(-10, 10, self.num_pts)
        self.input_std = np.linspace(0, 20, self.num_pts)
        # neuron model parameter
        self.ma =  MomentActivation()
        try:
            print('Loading table...')
            U, S, X = self.load_table()
        except:
            print('Loading failed! Generating new look-up table...')
            U, S, X = self.gen_table()
            
        self.interp_mean = RegularGridInterpolator( (self.input_mean , self.input_std), U.T)
        self.interp_std = RegularGridInterpolator( (self.input_mean , self.input_std), S.T)
        self.interp_chi = RegularGridInterpolator( (self.input_mean , self.input_std), X.T)
                
        
    
    def load_table(self):        
        dat = np.load(self.file)
        U = dat['U']
        S = dat['S']
        X = dat['X']
        return U, S, X
        
    
    def gen_table(self):
        '''generate look-up table on a square grid'''        
        xv, yv = np.meshgrid(self.input_mean, self.input_std, indexing='xy')
        xv = xv.flatten()
        yv = yv.flatten()
        
        U = self.ma.mean(xv,yv).reshape(self.num_pts,self.num_pts)
        S,_ = self.ma.std(xv,yv)
        S = S.reshape(self.num_pts,self.num_pts)
        X = self.ma.chi(xv,yv).reshape(self.num_pts,self.num_pts)
        
        np.savez(self.file, input_mean=self.input_mean, input_std=self.input_std, U=U, S=S, X=X)
        
        return U, S, X
        

if __name__=='__main__':
    
    solver = LookUpSolver()
    
    input_mean = np.linspace(-10,10,100)
    input_std = np.ones_like(input_mean)*1
    
    r = solver.interp_mean( (input_mean, input_std))
    
    U,S,X = solver.load_table()
    
    plt.close('all')
    plt.plot(input_mean, r)
    plt.plot(solver.input_mean, U[:,0],'.')
    
    #plt.imshow(U)
    
    
