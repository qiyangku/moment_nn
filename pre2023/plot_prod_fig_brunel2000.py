# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 15:45:13 2023

@author: dell
"""
import numpy as np
from matplotlib import pyplot as plt
import os
import fnmatch

def load_data(path, indx=None):    
    data_files = os.listdir(path)    
    if indx == None:
        meta_dat = np.load(path+'meta_data.npz', allow_pickle=True)
        return meta_dat        
    for f in data_files:
        if fnmatch.fnmatch( f, str(indx).zfill(3) + '*.npz'):
            dat = np.load(path+f, allow_pickle=True)
            return dat
#%%
#path = './runs/pre2023_brunel_delay_2023_oct_13/'    # has oscillation
#path = './runs/pre2023_brunel_delay_zero_2023_oct_14/'    # no oscillation
path = './runs/pre2023_brunel_delay_longer_2023_oct_14/' #delay = 0.5*tau, has oscillation
#path = './runs/pre2023_brunel_delay_03_2023_oct_14/'  #no oscillation

meta_dat = load_data(path) #load meta data

uext = meta_dat['uext_array']
ie_ratio = meta_dat['ie_ratio_array']

# analysis 
size = (len(uext), len(ie_ratio))
mean_pop_avg = np.zeros(size)
ff_pop_avg = np.zeros(size)
mean_pop_std = np.zeros(size)
ff_pop_std = np.zeros(size)

osc_amp = np.zeros(size)
osc_freq = np.zeros(size)

for i in range(size[0]):
    print('Processing... {}/{}'.format(i, size[0]))
    for j in range(size[1]):
        indx = np.ravel_multi_index((i,j), size )
        dat = load_data(path, indx)
        u = dat['mnn_mean']
        s = dat['mnn_std']
        ff = s**2/u
        
        # average over second half of simulation, to deal with oscillating solutions
        cut_off = int(u.shape[1]/2)
        u_time_avg = np.mean(u[:, cut_off:], axis = 1)  #average over time
        ff_time_avg = np.mean(ff[:, cut_off:], axis = 1) 
               
        # population stats
        mean_pop_avg[i,j] = np.mean(u_time_avg)
        ff_pop_avg[i,j] = np.mean(ff_time_avg)
        mean_pop_std[i,j] = np.std(u_time_avg)
        ff_pop_std[i,j] = np.std(ff_time_avg)
        
        # detect oscillation
        tmp = np.mean(u[:, cut_off:], axis=0) #population average, no time average
                
        if ie_ratio[j]>4: # no oscilation found for excitation dominant regime
            osc_amp[i,j] = 0.5*(np.max(tmp)-np.min(tmp)) #rough estimate of oscillation amplitude
            
            #if osc_amp[i,j]>1e-5:
            psd = np.abs(np.fft.fft(tmp))
            psd[0]=0
            psd = psd[:int(len(psd)/2)] #discard mirrored result
            osc_freq[i,j] = np.argmax(psd)/(config['T_mnn']*0.02)  # psd peak index * df, which is 1/simulation time (ms)
            

        
#        U_pop[i,j,:] = np.mean(u, axis=0)
#        S_pop[i,j,:] = np.mean(s, axis=0)
        
#        U_end[i,j,:] = u[:,-1]
#        S_end[i,j,:] = s[:,-1]

#U_sample[i,j,:,:] = u[:nsamples,::down_sample_ratio]
#S_sample[i,j,:,:] = s[:nsamples,::down_sample_ratio]
#%% 
plt.close('all')

extent = (ie_ratio[0], ie_ratio[-1], uext[0], uext[-1]  )   

# heat map
plt.figure()
plt.subplot(2,2,1)
plt.imshow(mean_pop_avg, origin = 'lower', extent=extent, aspect='auto')
plt.ylabel('External input rate (sp/ms)')
plt.colorbar()
plt.title('Pop. avg. firing rate (sp/ms)')

plt.subplot(2,2,2)
plt.imshow(ff_pop_avg, origin = 'lower', extent=extent, aspect='auto', cmap='plasma', vmin=0,vmax=1)
plt.title('Pop. avg. FF')
plt.colorbar()

plt.subplot(2,2,3)
plt.imshow(mean_pop_std, origin = 'lower', extent=extent, aspect='auto')
plt.ylabel('External input rate (sp/ms)')
plt.xlabel('Inh-to-ext ratio')
plt.colorbar()
plt.title('Pop. std. firing rate  (sp/ms)')

plt.subplot(2,2,4)
plt.imshow(ff_pop_std, origin = 'lower', extent=extent, aspect='auto', cmap='plasma', vmin=0,vmax=1)
plt.title('Pop. std. FF')
plt.xlabel('Inh-to-ext ratio')
plt.colorbar()

plt.tight_layout()


plt.figure()
plt.subplot(2,2,1)
plt.imshow(osc_amp, origin = 'lower', extent=extent, aspect='auto')
plt.colorbar()

plt.subplot(2,2,2)
plt.imshow(osc_freq, origin = 'lower', extent=extent, aspect='auto')
plt.colorbar()

#i,j = 2,3
#indx = np.ravel_multi_index((i,j), size )
indx = 89
dat = load_data(path, indx)
uu = dat['mnn_mean']
plt.subplot(2,2,3)
t = np.linspace(0, config['T_mnn'] , uu.shape[1])*20
plt.plot(t, np.mean(uu,axis=0) )  #population avg
plt.xlabel('Time (ms)')
plt.ylabel('Pop avg firing rate (sp/ms)')
plt.subplot(2,2,4)
plt.plot(t, uu[[0,5,10,22],:].T )  #invidual neuron
plt.xlabel('Time (ms)')
plt.ylabel('Firing rates (sp/ms)')
plt.tight_layout()

#%%
# line art
plt.close('all')
plt.subplot(2,2,1)
plt.plot(ie_ratio, mean_pop_avg.T*1e3)
plt.ylabel('Pop. avg. firing rate')

plt.subplot(2,2,2)
plt.plot(ie_ratio, ff_pop_avg.T)
plt.ylabel('Pop. avg. FF')

plt.subplot(2,2,3)
plt.plot(ie_ratio, mean_pop_std.T*1e3)
plt.ylabel('Pop. std. firing rate')

plt.subplot(2,2,4)
plt.plot(ie_ratio, ff_pop_std.T)
plt.ylabel('Pop. std. FF')


#%% Load post-analysis data and plot it
import matplotlib.pyplot as plt

def plot_post_analysis(dat):
    
    ie_ratio = dat['ie_ratio']
    uext = dat['uext']
    mean_pop_avg = dat['mean_pop_avg']
    mean_pop_avg = dat['mean_pop_avg']
    ff_pop_avg = dat['ff_pop_avg']
    mean_pop_std = dat['mean_pop_std']
    ff_pop_std = dat['ff_pop_std']
    osc_amp = dat['osc_amp']
    osc_freq = dat['osc_freq']
    
    plt.close('all')

    extent = (ie_ratio[0], ie_ratio[-1], uext[0], uext[-1]  )   

    # heat map
    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(mean_pop_avg, origin = 'lower', extent=extent, aspect='auto')
    plt.ylabel('External input rate (sp/ms)')
    plt.colorbar()
    plt.title('Pop. avg. firing rate (sp/ms)')

    plt.subplot(2,2,2)
    plt.imshow(ff_pop_avg, origin = 'lower', extent=extent, aspect='auto', cmap='plasma', vmin=0,vmax=1)
    plt.title('Pop. avg. FF')
    plt.colorbar()

    plt.subplot(2,2,3)
    plt.imshow(mean_pop_std, origin = 'lower', extent=extent, aspect='auto')
    plt.ylabel('External input rate (sp/ms)')
    plt.xlabel('Inh-to-ext ratio')
    plt.colorbar()
    plt.title('Pop. std. firing rate  (sp/ms)')

    plt.subplot(2,2,4)
    plt.imshow(ff_pop_std, origin = 'lower', extent=extent, aspect='auto', cmap='plasma', vmin=0,vmax=1)
    plt.title('Pop. std. FF')
    plt.xlabel('Inh-to-ext ratio')
    plt.colorbar()

    plt.tight_layout()
    
    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(osc_amp, origin = 'lower', extent=extent, aspect='auto')
    plt.colorbar()
    plt.xlabel('Inh-to-ext ratio')
    plt.ylabel('External input rate (sp/ms)')
    plt.title('Oscillation amplitude (sp/ms)')

    plt.subplot(2,2,2)
    plt.imshow(osc_freq, origin = 'lower', extent=extent, aspect='auto')
    plt.colorbar()
    plt.xlabel('Inh-to-ext ratio')
    plt.title('Oscillation frequency')

    plt.figure() #plot a slice
    i = np.where(uext==20)[0][0]    
    plt.subplot(2,2,1)
    plt.errorbar(ie_ratio, mean_pop_avg[i,:], mean_pop_std[i,:])   
    plt.ylabel('Pop. avg. firing rate (sp/ms)')
    plt.subplot(2,2,2)
    plt.errorbar(ie_ratio, ff_pop_avg[i,:], ff_pop_std[i,:])   
    plt.ylabel('Pop. avg. FF')
    
    plt.subplot(2,2,3)    
    plt.plot(ie_ratio, osc_amp[i,:])
    plt.ylabel('Oscillation amplitude (sp/ms)')
    
    

path = './runs/pre2023_brunel_delay_05_fine_2023_oct_14/'
dat = np.load(path+'post_analysis.npz')   
plot_post_analysis(dat)    

#%% Plot slice
path = './runs/pre2023_brunel_delay_05_slice_fine_2023_oct_16/'
dat = np.load(path+'post_analysis.npz')

ie_ratio = dat['ie_ratio']
uext = dat['uext']
mean_pop_avg = dat['mean_pop_avg']
mean_pop_avg = dat['mean_pop_avg']
ff_pop_avg = dat['ff_pop_avg']
mean_pop_std = dat['mean_pop_std']
ff_pop_std = dat['ff_pop_std']
osc_amp = dat['osc_amp']
osc_freq = dat['osc_freq']

crit_pts = [3.4, 6.4]

plt.close('all')
plt.figure() #plot a slice
plt.subplot(2,2,1)
#plt.errorbar(ie_ratio, mean_pop_avg[0,:], mean_pop_std[0,:])   
plt.fill_between(ie_ratio, mean_pop_avg[0,:]-mean_pop_std[0,:]/2, mean_pop_avg[0,:]+mean_pop_std[0,:]/2, alpha=0.3)
plt.plot(ie_ratio, mean_pop_avg[0,:])

plt.ylabel('Pop. avg. firing rate (sp/ms)')

for p in crit_pts:
    plt.plot([p,p],[-0.1,0.5],'--', color='gray')
plt.ylim([-0.05,0.5])

plt.subplot(2,2,2)
#plt.errorbar(ie_ratio, ff_pop_avg[0,:], ff_pop_std[0,:])   
plt.fill_between(ie_ratio, ff_pop_avg[0,:]-ff_pop_std[0,:]/2, ff_pop_avg[0,:]+ff_pop_std[0,:]/2, alpha=0.3) 
plt.plot(ie_ratio, ff_pop_avg[0,:])   
plt.ylabel('Pop. avg. FF')

for p in crit_pts:
    plt.plot([p,p],[-0.2,0.8],'--', color='gray')
plt.ylim([-0.2,0.8])



plt.subplot(2,2,3)    
plt.plot(ie_ratio, osc_amp[0,:])
plt.ylabel('Oscillation amplitude (sp/ms)')
plt.xlabel('Inh-to-ext ratio')

for p in crit_pts:
    plt.plot([p,p],[-0.001,0.008],'--', color='gray')
plt.ylim([-0.001,0.008])

plt.subplot(2,2,4)    
plt.plot(ie_ratio, osc_freq[0,:])
plt.ylabel('Oscillation frequency (Hz)')
plt.xlabel('Inh-to-ext ratio')

for p in crit_pts:
    plt.plot([p,p],[-1,15],'--', color='gray')
plt.ylim([-1,15])


plt.tight_layout()


#%% plot typical examples
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def plot_example(path, indx):
    dat = load_data(path, indx)
    
    uu = dat['mnn_mean']
    ff = dat['mnn_std']**2/dat['mnn_mean']
    config = dat['config'].item()
    
    #plt.close('all')
    plt.figure(figsize=(8.5,4))
    plt.subplot(2,3,1)
    t = np.linspace(0, config['T_mnn'] , uu.shape[1])*20
    plt.plot(t, np.mean(uu[:config['NE'],:],axis=0) )  #population avg (ext neuron, then inh)
    plt.plot(t, np.mean(uu[config['NE']:,:],axis=0) , '--')
    plt.xlabel('Time (ms)')
    plt.ylabel('Pop avg firing rate (sp/ms)')
    
    plt.subplot(2,3,2)    
    mean_u = np.mean(uu[:, int(uu.shape[1]/2):], axis = 1)
    mean_ff = np.mean(ff[:, int(uu.shape[1]/2):], axis = 1)    
    sorted_id = np.argsort(mean_u)     
    samples = np.arange(100,len(mean_u),3500)
    
    
    #copied color palette directly from https://carto.com/carto-colors/ 
    colors = ['#fbe6c5','#f5ba98','#ee8a82','#dc7176','#c8586c','#9c3f5d','#70284a']
    colors.reverse()
    colors=colors[::2]
    
    for i in range(len(samples)):    
        plt.plot(t, uu[sorted_id[samples[i]],:], color=colors[i])  #invidual neuron
    plt.xlabel('Time (ms)')
    plt.ylabel('Firing rates (sp/ms)')
    
    plt.subplot(2,3,3)
    plt.hist(mean_u[:config['NE']],50)
    plt.hist(mean_u[config['NE']:],50)
    plt.xlabel('Firing rate (sp/ms)')
    
    plt.subplot(2,3,4)
    plt.plot(t, np.mean(ff[:config['NE'],:], axis=0) )  #invidual neuron
    plt.plot(t, np.mean(ff[config['NE']:,:], axis=0) )  #invidual neuron
    plt.xlabel('Time (ms)')
    plt.ylabel('Pop avg Fano factor')
    
    plt.subplot(2,3,5)
    for i in range(len(samples)):    
        plt.plot(t, ff[sorted_id[samples[i]],:], color=colors[i])  #invidual neuron        
    plt.xlabel('Time (ms)')
    plt.ylabel('Fano factor')
    
    plt.subplot(2,3,6)
    plt.hist(mean_ff[:config['NE']],np.linspace(0,1,51))    
    plt.hist(mean_ff[config['NE']:],np.linspace(0,1,51))    
    plt.xlabel('Fano factor')
    
    plt.tight_layout()



#path = './runs/pre2023_brunel_delay_05_examples_2023_oct_16/'
path = './runs/pre2023_brunel_delay_05_slice_fine_2023_oct_16/'
meta_dat = np.load(path+'meta_data.npz', allow_pickle=True)
uext_array = meta_dat['uext_array']
ie_ratio_array = meta_dat['ie_ratio_array']

plt.close('all')
plot_example(path, 15)
plot_example(path, 17) #crit point 1
plot_example(path, 25)
plot_example(path, 32) #crit point 2
plot_example(path, 35)
# i = np.where(uext_array==20)[0][0]
# plt.close('all')
# for j in range(8):
#     #j = np.where(ie_ratio_array==jj)[0][0]
#     indx = np.ravel_multi_index((i,j),[len(uext_array), len(ie_ratio_array)])       
#     plot_example(path, indx)
    
#%% manually look into the dynamics

path = './runs/pre2023_brunel_delay_05_examples_2023_oct_16/'
dat = load_data(path, 7)

cut_off = 800
u = dat['mnn_mean'][:,cut_off:]
s = dat['mnn_std'][:,cut_off:]

plt.close('all')
plt.figure()
plt.subplot(2,2,1)
for k in [0,99,199]:
    plt.plot( u[k,:], s[k,:]**2)
    
plt.subplot(2,2,2)  
for k in [1, -1]:
    plt.plot( u[0,:], u[k,:])

plt.subplot(2,2,3)  
for k in range(1,7):
    plt.plot( s[0,:]**2, s[k,:]**2)
    
    