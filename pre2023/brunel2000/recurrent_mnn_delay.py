"""
Search through parameter space to identify different regimes

Async vs  sync: rho = 0 vs rho > 0 (<0?)
Regular vs irregular: FF -> 0 vs FF -> 1

Can I identify all these regimes?

Plan: 
    1, first I only need to sweep MNN - which is faster than SNN;
    2, next, simulate SNN by taking a slice


@author: Yang Qi
"""

#from mnn_core.rec_snn_simulator import *
#from mnn_core.rec_mnn_simulator import RecurrentMNN
from pre2023.brunel2000.rec_mnn_simulator import *
#from mnn_core.preprocessing import gen_synaptic_weight <-- this one scales weight by # of neurons...
from matplotlib import pyplot as plt

# parameter settings from Moreno-Bote (2014)

def gen_config(N=100, ie_ratio=4.0, uext=10.0): #generate config file
    w = 0.1    

    config = {
    'Vth': 20, #mV, firing threshold, default 20
    'Vres': 10, #mV reset potential; default 0
    'Tref': 2, #ms, refractory period, default 5
    'NE': int(0.8*N),
    'NI': int(0.2*N),
    'g': ie_ratio,     #I-E ratio
    'wee':{'mean': w, 'std': 1e-6},
    'wei':{'mean': -w*ie_ratio, 'std': 1e-6},
    'wie':{'mean': w, 'std': 1e-6},    
    'wii':{'mean': -w*ie_ratio, 'std': 1e-6},
    'uext': uext, # external firing rate kHz; rate*in-degree*weight = 0.01*1000*0.1 = 1 kHz
    #'wie':{'mean': 5.9, 'std': 0.0},    
    #'wii':{'mean': -9.4, 'std': 0.0},        
    'conn_prob': 0.1, #connection probability; N.B. high prob leads to poor match between mnn and snn
    'sparse_weight': False, #use sparse weight matrix; not necessarily faster but saves memory
    'randseed':None,
    'dT': 200, #ms spike count time window
    'delay': 0.1, # synaptic delay (uniform) in Brunel it's around 2 ms (relative to 20 ms mem time scale)
    'dt':0.02, # integration time step for mnn
    }

    return config


def para_sweep(pop_size, T_mnn = 10, save_results=False):
    '''Do a parameter sweep over the weight space'''
    
    #10 x 10 sweep can be done in 12 hrs; 30x30 in 2 days, for N=12500, no correlation, dt=0.02
    
    
    #uext = np.linspace(0.0 , 2.0 ,11)
    #ie_ratio = np.linspace(0.0 , 8.0 ,10)
    
    uext = np.linspace(0.0 , 40.0 ,5)[1:]  # u=0 is pointless
    ie_ratio = np.linspace(0.0 , 8.0 ,5)[1:]
    
    nsteps = 100 # down sample time steps to save
    nsamples = 10 # sample neurons
    
    U_pop = np.zeros( (len(uext), len(ie_ratio), nsteps) ) # population averaged 
    S_pop = np.zeros( (len(uext), len(ie_ratio), nsteps) )
    
    # need to save individual neurons as well.. firing rates are highly inhomogeneous!!
    U_end = np.zeros( (len(uext), len(ie_ratio), pop_size) )
    S_end = np.zeros( (len(uext), len(ie_ratio), pop_size) )
    
    # save full time course of a small number of neurons 
    U_sample = np.zeros( (len(uext), len(ie_ratio), nsamples, nsteps) )
    S_sample = np.zeros( (len(uext), len(ie_ratio), nsamples, nsteps) )
    
    #R = np.zeros( (len(uext), len(ie_ratio), pop_size, pop_size) ) #no correlation, too expensive to compute
    
    t0 = time.perf_counter()
    for i in range(len(uext)):
        for j in range(len(ie_ratio)):
            print('Starting iteration i={}/{}, j={}/{}'.format(i,len(uext),j,len(ie_ratio)))
            config = gen_config( N=pop_size, ie_ratio=ie_ratio[j], uext=uext[i] )
            W = gen_synaptic_weight(config) #doesn't take too much time with 1e4 neurons
            input_gen = InputGenerator(config)
            mnn_model = RecurrentMNN(config, W, input_gen)

            u,s= mnn_model.run_no_corr(T_mnn, record_ts = True)
            
            down_sample_ratio = int(u.shape[1]/nsteps)
            
            # save mean only not good enough. need to down sample time and neurons
            U_pop[i,j,:] = np.mean(u[:,::down_sample_ratio], axis=0)
            S_pop[i,j,:] = np.mean(s[:,::down_sample_ratio], axis=0)
            
            U_end[i,j,:] = u[:,-1]
            S_end[i,j,:] = s[:,-1]
            
            U_sample[i,j,:,:] = u[:nsamples,::down_sample_ratio]
            S_sample[i,j,:,:] = s[:nsamples,::down_sample_ratio]
            
            #R[i,j,:,:] = rho#[:,:,-1]
            #print('WE={}, ie_ratio={}'.format(WE[i],ie_ratio[j]))
            print('Time Elapsed: ', int(time.perf_counter() -t0) )
    if save_results:
        np.savez('rec_mnn_delay_para_sweep.npz', uext=uext, config=config, ie_ratio=ie_ratio, mnn_mean_pop=U_pop,mnn_std_pop=S_pop, \
                 mnn_mean_end = U_end, mnn_std_end=S_end, mnn_mean_sample=U_sample, mnn_std_sample=S_sample
                 )

    return uext, ie_ratio, U_pop, S_pop
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
    
    #config = gen_config( N=pop_size, ie_ratio=8.0, uext=40 ) #<-- highly heterogeneous rate and FF, even though network is homo!
    config = gen_config( N=pop_size, ie_ratio=3.0, uext=20 )
    # NB: uext are multiples of 0.01 in Brunel 2000; 
    # For me they need to be multiplied by the in-degree, so 10 kHz is the baseline
    
    W = gen_synaptic_weight(config) #doesn't take too much time with 1e4 neurons
    
    input_gen = InputGenerator(config)
    mnn_model = RecurrentMNN(config, W, input_gen)
    #snn_model = InteNFireRNN(config, W , input_gen)
    
    # simulate mnn
    t0 = time.perf_counter()
    #T_mnn = 1    #in practice needs 10 >> tau
    u,s = mnn_model.run_no_corr(T_mnn, record_ts = record_ts)
    print('Time elapsed (min): ', int(time.perf_counter()-t0)/60)
    return u, s
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

def plot_results():
    dat = np.load('rec_mnn_delay_para_sweep.npz', allow_pickle=True)
    uext = dat['uext']
    ie_ratio = dat['ie_ratio']
    mnn_mean = dat['mnn_mean']
    mnn_std = dat['mnn_std']
    mnn_corr = dat['mnn_corr']
    
    config = dat['config'].item()
    
    NE = config['NE']
    NI = config['NI']
    
    
    mean_E = np.mean(mnn_mean[:,:, :NE], axis=-1) #average over all neurons
    mean_I = np.mean(mnn_mean[:,:, NE:], axis=-1)
    std_E =  np.mean(mnn_std[:,:, :NE], axis=-1)
    std_I =  np.mean(mnn_std[:,:, NE:], axis=-1)
    
    corr_EE = (np.sum(np.sum(mnn_corr[:,:,:NE,:NE], axis=-1),axis=-1) - NE)/(NE*NE-NE)
    corr_EI = np.sum(np.sum(mnn_corr[:,:, :NE,NE:], axis=-1),axis=-1)/NE/NI
    corr_IE = np.sum(np.sum(mnn_corr[:,:, NE:,:NE], axis=-1),axis=-1)/NE/NI
    corr_II = (np.sum(np.sum(mnn_corr[:,:,NE:,NE:], axis=-1),axis=-1) - NI)/(NI*NI-NI)
    
    plt.close('all')
    plt.figure()    
    plt.subplot(2,2,1)
    plt.imshow(mean_E)
    plt.colorbar()
    plt.subplot(2,2,2)    
    plt.imshow(std_E**2/mean_E)
    plt.colorbar()
    
    plt.subplot(2,2,3)
    plt.imshow(mean_I)
    plt.colorbar()
    plt.subplot(2,2,4)
    plt.imshow(std_I**2/mean_I)
    plt.colorbar()
    
    
    
    #corr
    a = 1e-2
    
    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(corr_EE, vmax=a,vmin=-a, cmap = 'coolwarm')
    plt.title('E-E correlation')
    plt.subplot(2,2,2)
    plt.imshow(corr_EI, vmax=a,vmin=-a, cmap = 'coolwarm')
    plt.title('E-I correlation')    
    plt.subplot(2,2,3)
    plt.imshow(corr_IE, vmax=a,vmin=-a, cmap = 'coolwarm')
    plt.title('I-E correlation')
    plt.subplot(2,2,4)
    plt.imshow(corr_II, vmax=a,vmin=-a, cmap = 'coolwarm')
    plt.title('I-I correlation')
    
    
    
    return
    

if __name__=='__main__':
    
    u, s = single_run(pop_size=1250, T_mnn = 10, record_ts = True)

    #uext, ie_ratio, U, S = para_sweep(pop_size=12500, T_mnn=10, save_results=True)
    #uext, ie_ratio, U, S = para_sweep(pop_size=1250, T_mnn=10, save_results=True)
    
    print('\007')
#    plot_results()
#from matplotlib import pyplot as plt

#plt.plot(np.mean(u,axis=0))
#plt.imshow( np.mean(U, axis=-1) )