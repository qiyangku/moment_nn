"""
Created on Mon Jun 21 2023

For benchmarking accuracy of mnn for varying input correlation.

@author: Yang Qi
"""

from mnn_core.rec_snn_simulator import *
from mnn_core.rec_mnn_simulator import RecurrentMNN
from mnn_core.preprocessing import gen_synaptic_weight, InputGenerator


# parameter settings from Moreno-Bote (2014)

def gen_config(shared_noise_scale = 0.065, N=100): #generate config file
    
    tot_noise = 76.5e-3+5e-3

    config = {
    'Vth': 1, #mV, firing threshold, default 20
    'Tref': 5, #ms, refractory period, default 5
    'NE': int(0.8*N),
    'NI': int(0.2*N),
    'var_ind' : tot_noise*(1-shared_noise_scale), # per ms, independent noise strength      
    'var_shr' : tot_noise*shared_noise_scale, # per ms, shared noise strength     
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

m = 6
pop_size=30
noise_scale = np.linspace(0,1,m)[:-1]

mean_frate = np.zeros((2,m))
mean_corr = np.zeros((2,m))
snn_corr = np.zeros((  int(pop_size*(pop_size-1)/2) , m ))
mnn_corr = np.zeros((  int(pop_size*(pop_size-1)/2) , m ))

s = 0.1825/2

indx = np.triu_indices(pop_size,k=1)

t00 = time.perf_counter()
for i in range( len(noise_scale) ):
    print('Processing iteration ', i)
    config = gen_config( noise_scale[i] , pop_size )
    W = gen_synaptic_weight(config)
    input_gen = InputGenerator(config)
    snn_model = InteNFireRNN(config, W , input_gen)
    mnn_model = RecurrentMNN(config, W, input_gen)
    
    # simulate mnn
    t0 = time.perf_counter()
    T_mnn = 1    #in practice needs 10 >> tau
    u,_,rho = mnn_model.run(T_mnn,s, record_ts = False)
    
    mean_frate[0,i] = np.mean(u)
    mean_corr[0,i] = np.mean(rho[indx])
    mnn_corr[:,i] = rho[indx]
    
    # simulate snn
    T = 100e3 #in practice 10 s is minimum (50 spk count of 200 ms)
    t0 = time.perf_counter()
    SpkTime, V, t = snn_model.run( T, s)
    dT_snn = time.perf_counter() - t0
    spk_count = spk_time2count(SpkTime, T, binsize = config['dT'])
    
    corr_coef = np.corrcoef(spk_count)
    snn_corr[:,i] = corr_coef[indx]
    
    corr_coef[np.isnan(corr_coef)] = 0
    
    
    mean_frate[0,i] = np.mean(spk_count)/config['dT']
    mean_corr[1,i] = np.mean(corr_coef[indx])
    
    print('Total time elapsed (min): ', np.round((time.perf_counter()-t00)/60,2))

#%% compare corr coef of all neurons
for i in range(m):
    plt.plot(snn_corr[:,i], mnn_corr[:,i],'.')

#%% plotting routine ver 2

mean_spk_count = np.mean(spk_count, axis = 1)
var_spk_count = np.var(spk_count, axis = 1)    

corr_coef = np.corrcoef(spk_count)
corr_coef[np.isnan(corr_coef)] = 0

N = len(SpkTime)
x = np.arange(N)


plt.close('all')

plt.subplot(2,2,1)

tmp_spk = SpkTime
for i in range(len(tmp_spk)):    
    a = np.array(tmp_spk[i])
    tmp_spk[i] = a[a<1000] # keep only first 1000 ms data for plotting purposes

plt.eventplot(tmp_spk[:100])
#plt.eventplot(SpkTime[:config['NE']], color='r')
#plt.eventplot(SpkTime[config['NE']:], color='b', lineoffsets=config['NE'])
plt.xlabel('time (ms)')
plt.ylabel('neuron index')    
#plt.xlim(0,5e3)

plt.subplot(2,2,2)
plt.plot( U[:,-1]*1e3,mean_spk_count/config['dT']*1e3 ,'.')
plt.gca().set_aspect('equal')
plt.xlabel('moment nn')
plt.ylabel('spiking nn')
plt.title('mean firing rate (Hz)')

plt.subplot(2,2,3)
plt.plot( S[:,-1]**2/U[:,-1], var_spk_count/mean_spk_count , '.')
plt.gca().set_aspect('equal')
plt.xlabel('moment nn')
plt.ylabel('spiking nn')
plt.title('Fano factor')

plt.subplot(2,2,4)
indx = np.triu_indices( N, k=1 )
tmp_rho = R[:,:,-1]
plt.plot( tmp_rho[indx] , corr_coef[indx], '.')
plt.gca().set_aspect('equal')
plt.xlim([-1, 1])
plt.ylim([-1, 1])
#plt.imshow(corr_coef, vmin=-1, vmax=1, cmap = 'coolwarm')
print('Average corr coef (snn): ', np.round(np.mean(corr_coef[indx]),3))
print('Average corr coef (mnn): ', np.round(np.mean(tmp_rho[indx]),3))
