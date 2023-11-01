from pre2023.model_validation.validate_w_spiking_neuron import InteNFire
import numpy as np
import scipy as sp
import time
from mnn_core.maf import *
from matplotlib import pyplot as plt

def input_output_anlaysis(input_type):
    inf = InteNFire(num_neurons = 1000) #time unit: ms        
    #N = 31
    #u = np.linspace(-0.5,2.5,N)
    
    #u = np.linspace(-2.5,5.0,N)
    #u = np.arange(1.25,9.25,0.25)
    #N = len(u)
    
    #s = np.array([0,5])#np.ones(N)*1.5
    
    #try 25 iterations
    # let's say current=1, w=0.1 => input spikes = 10 kHz, assume 1000 synapses, then it's 10 Hz per synapse
    N = 26 
    u = np.linspace(-1,4,N)
    s = np.linspace(0,5,N)
    T = 1e3 # Set to None to use adaptive duration
    
    emp_u = np.zeros((u.size,s.size))
    emp_s = np.zeros((u.size,s.size))
    
    start_time = time.time()
    for i in range(u.size):
        for j in range(s.size):
            SpkTime, _, _ = inf.run(T = T, input_mean = u[i], input_std = s[j], input_type = input_type, show_message = False)
            emp_u[i,j], emp_s[i,j] = inf.empirical_maf(SpkTime)    
                    
            progress = (i*N + j +1)/N/N*100
            elapsed_time = (time.time()-start_time)/60
            print('Progress: {:.2f}%; Time elapsed: {:.2f} min'.format(progress, elapsed_time ))
            
    
    #maf = MomentActivation()
    
    maf_u = np.zeros((u.size,s.size))
    maf_s = np.zeros((u.size,s.size))
    
    for j in range(s.size):
        maf_u[:,j] = inf.maf.mean(u,s[j]*np.ones(u.size))
        maf_s[:,j], _ = inf.maf.std(u,s[j]*np.ones(u.size))
    
    return emp_u, emp_s, maf_u, maf_s, u, s

if __name__=='__main__':
    emp_u, emp_s, maf_u, maf_s, u, s = input_output_anlaysis(input_type = 'spike')
    
    #maybe I should increase the # of neurons (# trials), and decrease simulation time not 10s but 2s is enougth?
    #so better parallelization and speed
    #omg this snn simulation takes forever, thanks to dt=0.001 ms. So 10^6 steps needed per 1 s simulation
    
    np.savez('./runs/benchmark_acc_maf.npz', emp_u=emp_u, emp_s=emp_s, maf_u=maf_u, maf_s=maf_s, u=u, s=s)


# #%% 2h - 5x5
# plt.close('all')
# plt.figure()
# plt.plot(emp_u.flatten(),maf_u.flatten(),'.')
# plt.plot([0, 0.1],[0, 0.1],'--')

# plt.figure()
# plt.plot(emp_s.flatten(),maf_s.flatten(),'.')
# plt.plot([0, 0.1],[0, 0.1],'--')

# #%%
# plt.close('all')

# # mean map
# plt.figure(figsize=(15,4))
# plt.subplot(1,3,1)
# plt.imshow(emp_u.T, origin = 'lower', extent = (u[0],u[-1],s[0],s[-1]) )
# plt.colorbar()

# plt.subplot(1,3,2)
# plt.imshow(maf_u.T, origin = 'lower')
# plt.colorbar()

# plt.subplot(1,3,3)
# plt.imshow(maf_u.T-emp_u.T, origin = 'lower')
# plt.colorbar()

# # variance map
# plt.figure(figsize=(15,4))
# plt.subplot(1,3,1)
# plt.imshow(emp_s.T, origin = 'lower', extent = (u[0],u[-1],s[0],s[-1]) )
# plt.colorbar()

# plt.subplot(1,3,2)
# plt.imshow(maf_s.T, origin = 'lower')
# plt.colorbar()

# plt.subplot(1,3,3)
# plt.imshow(maf_s.T-emp_s.T, origin = 'lower')
# plt.colorbar()

