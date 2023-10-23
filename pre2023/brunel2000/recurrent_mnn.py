"""
Created on Mon Jun 21 2023

For benchmarking accuracy of mnn for varying input correlation.

@author: Yang Qi
"""

#from mnn_core.rec_snn_simulator import *
#from mnn_core.rec_mnn_simulator import RecurrentMNN
from pre2023.brunel2000.rec_mnn_simulator import *
from mnn_core.preprocessing import gen_synaptic_weight#, InputGenerator
from matplotlib import pyplot as plt

# parameter settings from Moreno-Bote (2014)

def gen_config(N=100): #generate config file
    
    config = {
    'Vth': 20, #mV, firing threshold, default 20
    'Vres': 10, #mV reset potential; default 0
    'Tref': 2, #ms, refractory period, default 5
    'NE': int(0.8*N),
    'NI': int(0.2*N),
    'g': 1,     #I-E ratio
    'wee':{'mean': 1, 'std': 0.1},
    'wei':{'mean': -4, 'std': 0.1},
    'wie':{'mean': 1, 'std': 0.1},    
    'wii':{'mean': -4, 'std': 0.1},
    'uext': 1.0, # external firing rate kHz; rate*in-degree*weight = 0.01*1000*0.1 = 1 kHz
    #'wie':{'mean': 5.9, 'std': 0.0},    
    #'wii':{'mean': -9.4, 'std': 0.0},        
    'conn_prob': 0.1, #connection probability; N.B. high prob leads to poor match between mnn and snn
    'sparse_weight': True, #use sparse weight matrix; not necessarily faster but saves memory
    'randseed':None,
    'dT': 200, #ms spike count time window
    }

    return config

class InputGenerator():
    def __init__(self, config):
        self.NE = config['NE']
        self.NI = config['NI']
        self.N = config['NE']+config['NI']
        self.uext = config['uext']
        #define external input mean
        #
        self.input_mean = self.uext*np.ones((self.N,1)) 
        
        #calculate external input cov (assume independent Poisson spikes)
        self.input_cov = self.uext*np.eye(self.N)
        
        #self.L_ext = np.linalg.cholesky(self.input_cov)        
        
        return
    
    # def gen_gaussian_current(self, s, dt):     
    #     u = self.input_mean(s)
    #     z = np.matmul(self.L_ext, np.random.randn(self.N,1))        
    #     I_ext = u*dt + z*np.sqrt(dt)
    #     return I_ext

pop_size=1250 #doesn't take too much time

#mean_frate = np.zeros((2,m))
#mean_corr = np.zeros((2,m))
#snn_corr = np.zeros((  int(pop_size*(pop_size-1)/2) , m ))
#mnn_corr = np.zeros((  int(pop_size*(pop_size-1)/2) , m ))

config = gen_config( pop_size )
W = gen_synaptic_weight(config) #doesn't take too much time with 1e4 neurons

input_gen = InputGenerator(config)
mnn_model = RecurrentMNN(config, W, input_gen)
#snn_model = InteNFireRNN(config, W , input_gen)

# simulate mnn
#t0 = time.perf_counter()
T_mnn = 10    #in practice needs 10 >> tau
u,_,rho = mnn_model.run(T_mnn, record_ts = True)

#mean_frate[0,i] = np.mean(u)
#mean_corr[0,i] = np.mean(rho[indx])
#mnn_corr[:,i] = rho[indx]

# # simulate snn
# T = 100e3 #in practice 10 s is minimum (50 spk count of 200 ms)
# t0 = time.perf_counter()
# SpkTime, V, t = snn_model.run( T, s)
# dT_snn = time.perf_counter() - t0
# spk_count = spk_time2count(SpkTime, T, binsize = config['dT'])

# corr_coef = np.corrcoef(spk_count)
# snn_corr[:,i] = corr_coef[indx]

# corr_coef[np.isnan(corr_coef)] = 0

# mean_frate[0,i] = np.mean(spk_count)/config['dT']
# mean_corr[1,i] = np.mean(corr_coef[indx])

#print('Total time elapsed (min): ', np.round((time.perf_counter()-t00)/60,2))

