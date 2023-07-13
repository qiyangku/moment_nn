# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 21:19:12 2023

for plotting production figure

@author: Yang Qi
"""

import numpy as np
import matplotlib.pyplot as plt

dat = np.load('benchmark_pop_size.npz')
dat2 = np.load('benchmark_pop_size_without_corr.npz')

pop_size = dat['pop_size']
dT_snn = dat['dT_snn']
dT_mnn = dat['dT_mnn']
dT_mnn_no_corr = dat2['dT_mnn']

tmp = np.array([5e2, 1e4])


#scale y-axis for better visibility
scale1 = 10 # SNN simulation time = 1 s, dt = 0.1 ms (1e4 steps) => adjust to 10 s, (1e5 steps)
sclae2 = 2 # MNN simulation time 1, dt = 0.1 (10 steps) => adjust to 10, dt = 0.5 (20 steps)



c1 = '#085a65'
c2 = '#d14627'
c3 = '#307c78'

plt.close('all')
plt.figure(figsize = (3.5,3))
plt.loglog(pop_size,  scale1*dT_snn/60,'.', color = c1)
plt.loglog(pop_size, sclae2*dT_mnn/60,'.', color = c2)
plt.loglog(pop_size, sclae2*dT_mnn_no_corr/60,'.', color=c3)
plt.loglog(tmp, 1e-6*tmp**2, color = c1)
plt.loglog(tmp, 6e-11*tmp**3, color = c2)
plt.loglog(tmp, 1e-9*tmp**2, color=c3)
plt.legend(['SNN', 'MNN', 'MNN (no corr)'], frameon=False)
plt.ylabel('CPU time (min)')
plt.xlabel('Population size')
plt.tight_layout()
#plt.gca().set_ylim(1e-5,3e1)
#np.savez('benchmark_pop_size_20230629.npz', pop_size=pop_size, dT_snn=dT_snn, dT_mnn=dT_mnn, config=config, mean_frate=mean_frate)


#%% speed comparison between different methods
from pre2023.utils import *
import matplotlib.pyplot as plt

dat = np.load('benchmark_speed_comparison.npz')
dT = dat['dT']
x = dat['label']

plt.close('all')
plt.figure(figsize=(3.5,3))

y = dT.reshape(dat['n']**2, 8)

z = {r'MA $\mu$': y[:,0], r'MA $\sigma$': y[:,1], r'MA $\chi$':y[:,2], r'DI $\mu$':y[:,5], \
           r'DI $\sigma$': y[:,6], r'DI $\chi$':y[:,7], 'FPE':y[:,3], 'Look-up':y[:,4]  }

plt.boxplot(z.values(), labels = z.keys(),  sym='' )
plt.yscale('log')
plt.ylabel(r'CPU time ($\mu s$)')
plt.xticks(rotation = 30) 
plt.tight_layout()


#%% benchmark of different components of MA
from pre2023.utils import *
import matplotlib.pyplot as plt

dat = np.load('benchmark_speed.npz')

input_mean = dat['input_mean']
input_std = dat['input_std']
dT = dat['dT']

plt.close('all')

# check the range of data
#tmp = np.median(dT[:,:,2,:], axis=-1)
#plt.hist(tmp.flatten(),50)
##%%

vmax = [46.5, 103.5, 8]
vmin = [44, 100, 7.6]

extent = [input_mean[0], input_mean[-1], input_std[0], input_std[-1]]
cmap = ['inferno','inferno','inferno']
title = ['Mean firing rate', 'Firing variability', 'Linear res. coef.']

for i in range(3):
    plt.figure(figsize=(4,3))
    img = np.median(dT[:,:,i,:], axis=-1)
    img = medianFilter(img) # remove shot noise
    plt.imshow( img, origin = 'lower', extent=extent, vmin = vmin[i], vmax = vmax[i], cmap =cmap[i]) #unit: ms
    
    # if i==0:        
    #     plot_boundary(10)
    #     plot_boundary(-10)
    #     plot_boundary(6)
    #     #plot_boundary(4.5)
    
    plt.colorbar(label = r'CPU time ($\mu s$)')
    plt.xlabel(r'Input current mean $\bar{\mu}$')
    plt.ylabel(r'Input current std $\bar{\sigma}$')    
    plt.title(title[i])
    plt.tight_layout()
    

#%% plot MA components; heat map
import matplotlib.pyplot as plt
from mnn_core.maf import MomentActivation

n = 100
input_mean = np.linspace(-10,10,n)
input_std = np.linspace(0,20,n)

ma = MomentActivation()

xv, yv = np.meshgrid(input_mean, input_std, indexing='xy')
z = np.zeros((n,n,3))

z[:,:,0] = ma.mean( xv.flatten(), yv.flatten() ).reshape(n,n)
z[:,:,1] = ma.std( xv.flatten(), yv.flatten() )[0].reshape(n,n)
z[:,:,2] = ma.chi( xv.flatten(), yv.flatten() ).reshape(n,n)

extent = [input_mean[0],input_mean[-1],input_std[0],input_std[-1]  ] 

label = [r'Mean firing rate (sp/ms)',r'Firing variability (sp/ms$^{1/2}$)',r'Linear res. coef.']
         

plt.close('all')
for i in range(3):
    plt.figure(figsize = (3.5,3))
    plt.imshow(z[:,:,i], extent=extent,  origin = 'lower', cmap = 'cividis', interpolation = 'lanczos')    
    plt.colorbar()#label = label[i])
    plt.contour(input_mean, input_std, z[:,:,i], 8, colors=['k'], antialiased=True, linestyles='dotted')
    plt.gca().set_position((0.1,0.2,0.65,0.65))
    plt.title(label=label[i])
    plt.xlabel(r'Input current mean $\bar{\mu}$')
    plt.ylabel(r'Input current std $\bar{\sigma}$')   
    #plt.tight_layout()


#%% plot linear Fisher info vs pop size; mean, std, corr
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

plt.close('all')

for fname in ['fisher_info_vs_N_cov_uniform.npz', 'fisher_info_vs_N_cov_cosine.npz']:

    dat = np.load(fname)
   
    
    N=dat['N']
    LFI=dat['LFI']
    
    
    # plot fisher information
    shr_noise_lvl = dat['frac_shr_noise']
    
    plt.figure(figsize=(3.5,3))
    
    alpha = 1-np.arange(LFI.shape[1])*0.16
    
    for i in range(LFI.shape[1]):
        z = signal.savgol_filter(LFI[1:,i], 9, 3)
        plt.loglog(N[1:],  z , color = '#c01f25', alpha= alpha[i])
        plt.gca().set_position((0.25,0.2,0.7,0.7))
    
    plt.ylabel(r'Information rate (ms$^{-1}$)')
    plt.xlabel('Population size')
    plt.legend( ['c = {}'.format(k) for k in np.round(shr_noise_lvl,1)  ])
    
    #plt.tight_layout()
    
    
    # plot mean and var
    
    plt.figure(figsize = (3.5,3))
    if fname == 'fisher_info_vs_N_cov_uniform.npz':
        plt.plot( dat['output_mean'][:800,-1] , dat['output_std'][:800,-1]**2,'.') # rho = 0.4
        print('noise level: ', shr_noise_lvl[-1])
        ff = dat['output_std'][:800,-1]**2/dat['output_mean'][:800,-1]
        print('max fano factor = ', np.max(ff))
        print('min fano factor = ', np.min(ff))
    else:
        plt.plot( dat['output_mean'][:800,2] , dat['output_std'][:800,2]**2,'.') # rho = 0.4
        print('noise level: ', shr_noise_lvl[2])
        
    plt.gca().set_position((0.25,0.2,0.7,0.7))
    plt.xlabel(r'Mean firing rate $\mu$ (sp/ms)')
    plt.ylabel(r'Firing variability $\sigma^2$ (sp$^2$/ms)')
    #plt.tight_layout()
    
    
    # plot correlation
    plt.figure(figsize = (3.5,3))
    dn = 10 #down-sampling
    
    #plt.subplot(1,2,1)    
    # C_in = dat['input_cov'][::dn,::dn,-1]
    # C_in = C_in[:80,:80] #only excitatory neurons
    # s_in = np.sqrt(np.diag(C_in))
    # corr_in = C_in/s_in.reshape(-1,1)/s_in.reshape(1,-1)
    # plt.imshow(corr_in, vmin=-1,vmax=1, cmap = 'coolwarm')
    # plt.colorbar()
    # plt.tight_layout()
    # plt.xticks([])
    # plt.yticks([])
    
    corr_out = dat['output_corr'][::dn,::dn,-1]
    corr_out = corr_out[:80,:80]
    plt.imshow(corr_out,vmin=-0.8,vmax=0.8, cmap = 'coolwarm', interpolation='none')
    plt.colorbar()
    plt.gca().set_position((0.02,0.2,0.7,0.7))
    
    #plt.tight_layout()
    plt.xticks([])
    plt.yticks([])
    






#%% plot mean, std, corr coef for mnn vs snn

#np.savez('recurrent_mnn_vs_snn_example.npz', spk_count=spk_count, config=config, stim=s, mnn_mean=U,mnn_std=S,mnn_corr=R, snn_T_ms=T)

dat = np.load('recurrent_mnn_vs_snn_example.npz', allow_pickle=True)

spk_count=dat['spk_count']
U=dat['mnn_mean']
S=dat['mnn_std']
R=dat['mnn_corr']
config = dat['config'].item()

mean_spk_count = np.mean(spk_count, axis = 1)
var_spk_count = np.var(spk_count, axis = 1)    

corr_coef = np.corrcoef(spk_count)
corr_coef[np.isnan(corr_coef)] = 0

N = spk_count.shape[0]
x = np.arange(N)

plt.close('all')


plt.subplot(1,3,1)
plt.plot( U[:,-1]*1e3, mean_spk_count/config['dT']*1e3 ,'.')
tmp = U[:,-1]*1e3
tmp = [np.min(tmp), np.max(tmp)]
plt.plot( tmp, tmp ,'-r')
plt.gca().set_aspect('equal')
plt.xlabel('moment nn')
plt.ylabel('spiking nn')
plt.title('mean firing rate (Hz)')


# Fano factor is cool => wide range from <1 to >1
plt.subplot(1,3,2)
plt.plot( S[:,-1]**2/U[:,-1], var_spk_count/mean_spk_count , '.')
#plt.plot( S[:,-1]**2, var_spk_count/config['dT'] , '.')
tmp = S[:,-1]**2/U[:,-1]
#tmp = S[:,-1]**2
tmp = [np.min(tmp), np.max(tmp)]
plt.plot( tmp, tmp ,'-r')
plt.gca().set_aspect('equal')
plt.xlabel('moment nn')
plt.ylabel('spiking nn')
plt.title('Fano factor')
#plt.title('Firing variability')


#>>> to be replaced by average for differnt input corr
plt.subplot(1,3,3)
indx = np.triu_indices( N, k=1 )
tmp_rho = R[:,:,-1]
plt.plot( tmp_rho[indx] , corr_coef[indx], '.')
plt.gca().set_aspect('equal')
plt.xlim([-0.5, 0.5])
plt.ylim([-0.5, 0.5])
#plt.imshow(corr_coef, vmin=-1, vmax=1, cmap = 'coolwarm')
print('Average corr coef (snn): ', np.round(np.mean(corr_coef[indx]),3))
print('Average corr coef (mnn): ', np.round(np.mean(tmp_rho[indx]),3))

plt.tight_layout()

