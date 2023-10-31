"""
Created on Mon Jun 21 2023

For benchmarking accuracy of mnn for varying input correlation.

@author: Yang Qi
"""

#from mnn_core.rec_snn_simulator import *
#from mnn_core.rec_mnn_simulator import RecurrentMNN
from pre2023.brunel2000.rec_mnn_simulator import *
from mnn_core.preprocessing import gen_synaptic_weight
from matplotlib import pyplot as plt

# parameter settings from Moreno-Bote (2014)

def gen_config(N=100, ie_ratio=4.0, uext=1.0): #generate config file
    
    config = {
    'Vth': 20, #mV, firing threshold, default 20
    'Vres': 10, #mV reset potential; default 0
    'Tref': 2, #ms, refractory period, default 5
    'NE': int(0.8*N),
    'NI': int(0.2*N),
    'g': 1,     #I-E ratio
    'wee':{'mean': 0.1, 'std': 0.01},
    'wei':{'mean': -0.1*ie_ratio, 'std': 0.01},
    'wie':{'mean': 0.1, 'std': 0.01},    
    'wii':{'mean': -0.1*ie_ratio, 'std': 0.01},
    'uext': uext, # external firing rate kHz; rate*in-degree*weight = 0.01*1000*0.1 = 1 kHz
    #'wie':{'mean': 5.9, 'std': 0.0},    
    #'wii':{'mean': -9.4, 'std': 0.0},        
    'conn_prob': 0.2, #connection probability; N.B. high prob leads to poor match between mnn and snn
    'sparse_weight': True, #use sparse weight matrix; not necessarily faster but saves memory
    'randseed':None,
    'dT': 200, #ms spike count time window
    }

    return config




def para_sweep(pop_size=2000, T_mnn = 10, save_results=False):
    '''Do a parameter sweep over the weight space'''
    uext = np.linspace(0.0 , 2.0 ,11)
    ie_ratio = np.linspace(0.0 , 8.0 ,10)
    
    U = np.zeros( (len(uext), len(ie_ratio), pop_size) )
    S = np.zeros( (len(uext), len(ie_ratio), pop_size) )
    R = np.zeros( (len(uext), len(ie_ratio), pop_size, pop_size) )
    
    t0 = time.perf_counter()
    for i in range(len(uext)):
        for j in range(len(ie_ratio)):
            print('Starting iteration i={}/{}, j={}/{}'.format(i,len(uext),j,len(ie_ratio)))
            config = gen_config( N=pop_size, ie_ratio=ie_ratio[j], uext=uext[i] )
            W = gen_synaptic_weight(config) #doesn't take too much time with 1e4 neurons
            input_gen = InputGenerator(config)
            mnn_model = RecurrentMNN(config, W, input_gen)

            u,s,rho = mnn_model.run(T_mnn, record_ts = False)
            U[i,j,:] = u.flatten()#[:,-1]
            S[i,j,:] = s.flatten()#[:,-1]
            R[i,j,:,:] = rho#[:,:,-1]
            #print('WE={}, ie_ratio={}'.format(WE[i],ie_ratio[j]))
            print('Time Elapsed: ', int(time.perf_counter() -t0) )
    if save_results:
        np.savez('brunel2000_para_sweep.npz', uext=uext, config=config, ie_ratio=ie_ratio, mnn_mean=U,mnn_std=S,mnn_corr=R)

    return uext, ie_ratio, U, S, R    
    # def gen_gaussian_current(self, s, dt):     
    #     u = self.input_mean(s)
    #     z = np.matmul(self.L_ext, np.random.randn(self.N,1))        
    #     I_ext = u*dt + z*np.sqrt(dt)
    #     return I_ext

def single_run(pop_size=1250, T_mnn = 1, record_ts = False ):
     #doesn't take too much time
    
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
    t0 = time.perf_counter()
    #T_mnn = 1    #in practice needs 10 >> tau
    u,_,rho = mnn_model.run(T_mnn, record_ts = record_ts)
    print('Time elapsed (min): ', int(time.perf_counter()-t0)/60)
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

if __name__=='__main__':
    single_run(pop_size=2000, T_mnn = 1)
    #uext, ie_ratio, U, S, R = para_sweep(pop_size=1250, T_mnn=1, save_results=True)
    
#%%

from matplotlib import pyplot as plt

plt.imshow(np.mean(U,axis=-1))
#plt.plot(uext, np.mean(U,axis=-1))