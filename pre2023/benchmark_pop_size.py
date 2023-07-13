"""
Created on Mon Jun 21 2023

For benchmarking efficiency of mnn vs snn as network size increases.

@author: Yang Qi
"""

from mnn_core.rec_snn_simulator import *
from mnn_core.rec_mnn_simulator import RecurrentMNN
from mnn_core.preprocessing import gen_synaptic_weight, InputGenerator


# parameter settings from Moreno-Bote (2014)

def gen_config(N): #generate config file
    
    config = {
    'Vth': 1, #mV, firing threshold, default 20
    'Tref': 5, #ms, refractory period, default 5
    'NE': int(0.8*N),
    'NI': int(0.2*N),
    'var_ind' : 76.5e-3, # per ms, independent noise strength      
    'var_shr' : 5e-3, # per ms, shared noise strength     
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
    }

    return config

m = 21
pop_size = np.round(np.logspace(1,4,m))
mean_frate = np.zeros(m)

dT_mnn = np.zeros(m)
dT_snn = np.zeros(m)
dT_mnn_no_corr = np.zeros(m)

s = 0.1825/2

t00 = time.perf_counter()
for i in range( len(pop_size) ):
    print('Processing iteration ', i)
    config = gen_config( pop_size[i] )
    W = gen_synaptic_weight(config)
    input_gen = InputGenerator(config)
    snn_model = InteNFireRNN(config, W , input_gen)
    mnn_model = RecurrentMNN(config, W, input_gen)
    
    # simulate mnn
    t0 = time.perf_counter()
    T_mnn = 1    #in practice needs 10 >> tau
    u,_,_ = mnn_model.run(T_mnn,s, record_ts = False)
    dT_mnn[i] = time.perf_counter() - t0
    print('Elapsed time for mnn: ', dT_mnn[i])
    
    # simulate mnn without corr
    t0 = time.perf_counter()
    u,_ = mnn_model.run_no_corr(T_mnn,s, record_ts = False)
    dT_mnn_no_corr[i] = time.perf_counter() - t0
    print('Elapsed time for mnn: ', dT_mnn_no_corr[i])
    
    mean_frate[i] = np.mean(u)
    
    # simulate snn
    T = 1e3 #in practice 10 s is minimum (50 spk count of 200 ms)
    t0 = time.perf_counter()
    SpkTime, V, t = snn_model.run( T, s)
    dT_snn[i] = time.perf_counter() - t0
    #spk_count = spk_time2count(SpkTime, T, binsize = config['dT'])
    print('Elapsed time for snn: ', dT_snn[i])
    
    print('Total time elapsed (min): ', np.round((time.perf_counter()-t00)/60,2))

#%% plot SNN vs MNN

plt.close('all')
plt.loglog(pop_size,dT_snn/60,'.b')
plt.loglog(pop_size,dT_mnn/60,'.r')
plt.loglog(pop_size,dT_mnn_no_corr/60,'.k')
plt.loglog(pop_size, 1e-7*pop_size**2,'b')
plt.loglog(pop_size, 1e-11*pop_size**3,'r')
plt.loglog(pop_size, 5e-10*pop_size**2,'k')
plt.legend(['SNN (1 s)', 'MNN', 'MNN (no corr)'])
plt.ylabel('CPU time (min)')
plt.xlabel('Population size')
plt.gca().set_ylim(1e-5,3e1)
#np.savez('benchmark_pop_size_20230629.npz', pop_size=pop_size, dT_snn=dT_snn, dT_mnn=dT_mnn, config=config, mean_frate=mean_frate)

#%% plot SNN vs MNN without corr

plt.close('all')
plt.loglog(pop_size,dT_snn/60,'.b')
plt.loglog(pop_size,dT_mnn/60,'.r')
tmp = np.array([1e2 ,1e5])
plt.loglog(tmp, 1e-7*tmp**2,'b')
plt.loglog(tmp, 5e-10*tmp**2,'r')
#plt.plot(pop_size,dT_snn/60,'.',pop_size,dT_mnn/60,'.')
plt.legend(['SNN (1 s)', 'MNN (no corr)'])
plt.ylabel('CPU time (min)')
plt.xlabel('Population size')
plt.gca().set_ylim(1e-5,1e2)
#np.savez('benchmark_pop_size_without_corr.npz', pop_size=pop_size, dT_snn=dT_snn, dT_mnn=dT_mnn, config=config, mean_frate=mean_frate)

#%%

N = 1000

config = {
'Vth': 1, #mV, firing threshold, default 20
'Tref': 5, #ms, refractory period, default 5
'NE': int(0.8*N),
'NI': int(0.2*N),
'var_ind' : 76.5e-3, # per ms, independent noise strength      
'var_shr' : 5e-3, # per ms, shared noise strength     
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
}

W = gen_synaptic_weight(config)
input_gen = InputGenerator(config)
mnn_model = RecurrentMNN(config, W, input_gen)
mnn_model.dt = 0.5
U,_,_ = mnn_model.run(10, 0.2, record_ts = True)