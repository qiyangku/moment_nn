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
        config = dat['config'].item()
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
    
    try:
        # load excitatory neuron stats
        mean_pop_avg = dat['mean_pop_avg'][:,:,0]
        mean_pop_avg = dat['mean_pop_avg'][:,:,0]
        ff_pop_avg = dat['ff_pop_avg'][:,:,0]
        mean_pop_std = dat['mean_pop_std'][:,:,0]
        ff_pop_std = dat['ff_pop_std'][:,:,0]
        osc_amp = dat['osc_amp']
        osc_freq = dat['osc_freq']
    except: # compatibility to older versions
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
    
    try:
        corr_pop_avg = dat['corr_pop_avg']
        plt.figure()
        
        plt.subplot(2,2,1)
        plt.imshow(corr_pop_avg[:,:,0], origin = 'lower', extent=extent, aspect='auto', cmap = 'coolwarm', vmin=-1,vmax=1)
        plt.xlabel('Inh-to-ext ratio')
        plt.ylabel('External input rate (sp/ms)')
        plt.title('E-E pops. correlation')
        plt.colorbar()
        
        plt.subplot(2,2,2)
        plt.imshow(corr_pop_avg[:,:,2], origin = 'lower', extent=extent, aspect='auto', cmap = 'coolwarm', vmin=-1,vmax=1)
        plt.xlabel('Inh-to-ext ratio')
        plt.ylabel('External input rate (sp/ms)')
        plt.title('E-I pops. correlation')
        plt.colorbar()
        
        plt.subplot(2,2,3)
        plt.imshow(corr_pop_avg[:,:,1], origin = 'lower', extent=extent, aspect='auto', cmap = 'coolwarm', vmin=-1,vmax=1)
        plt.xlabel('Inh-to-ext ratio')
        plt.ylabel('External input rate (sp/ms)')
        plt.title('I-I pops. correlation')
        plt.colorbar()
        
        plt.tight_layout()
        
    except:
        pass
    

#path = './runs/pre2023_brunel_delay_05_fine_2023_oct_14/'
path = './runs/pre2023_small_network_with_corr_fine_2023_nov_21/'

dat = np.load(path+'post_analysis.npz')   
plot_post_analysis(dat)    

#%% Plot slice (large network, no correlation)
path = './runs/pre2023_brunel_delay_05_slice_fine_2023_oct_16/'

dat = np.load(path+'post_analysis.npz')

ie_ratio = dat['ie_ratio']
uext = dat['uext']
mean_pop_avg = dat['mean_pop_avg']
ff_pop_avg = dat['ff_pop_avg']
mean_pop_std = dat['mean_pop_std']
ff_pop_std = dat['ff_pop_std']
osc_amp = dat['osc_amp']
osc_freq = dat['osc_freq']
osc_amp_ff = dat['osc_amp_ff']
mean_quartiles = dat['mean_quartiles']
ff_quartiles = dat['ff_quartiles']

crit_pts = [3.4, 6.4]

plt.close('all')
plt.figure() #plot a slice
plt.subplot(2,2,1)
#plt.errorbar(ie_ratio, mean_pop_avg[0,:], mean_pop_std[0,:])   
plt.fill_between(ie_ratio, mean_pop_avg[0,:]-mean_pop_std[0,:], mean_pop_avg[0,:]+mean_pop_std[0,:], alpha=0.3)
#plt.fill_between(ie_ratio, mean_quartiles[0,:,0], mean_quartiles[0,:,1], alpha=0.3)
plt.plot(ie_ratio, mean_pop_avg[0,:])

plt.ylabel('Pop. avg. firing rate (sp/ms)')

for p in crit_pts:
    plt.plot([p,p],[-0.1,0.5],'--', color='gray')
plt.ylim([-0.05,0.5])

plt.subplot(2,2,2)
#plt.errorbar(ie_ratio, ff_pop_avg[0,:], ff_pop_std[0,:])   
plt.fill_between(ie_ratio, ff_pop_avg[0,:]-ff_pop_std[0,:], ff_pop_avg[0,:]+ff_pop_std[0,:], alpha=0.3) 
#plt.fill_between(ie_ratio, ff_quartiles[0,:,0], ff_quartiles[0,:,1], alpha=0.3) 
plt.plot(ie_ratio, ff_pop_avg[0,:])   
plt.ylabel('Pop. avg. Fano factor')

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
    plt.plot([p,p],[-1,30],'--', color='gray')
plt.ylim([-1,30])


plt.tight_layout()


#%% plot typical examples
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def plot_example(path, indx):
    dat = load_data(path, indx)
    config = dat['config'].item()
    
    uu = dat['mnn_mean']
    ff = dat['mnn_std']**2/dat['mnn_mean']
    
    #consider only excitatory neurons
    uu = uu[:config['NE'],:]
    ff = ff[:config['NE'],:]
    
    #plt.close('all')
    plt.figure(figsize=(8.5,3.5))
    plt.subplot(2,3,1)
    t = np.linspace(0, config['T_mnn'] , uu.shape[1])*0.02
    
    t_len = int(uu.shape[1]/5)
    t_downsample_ratio = int(t_len/500)
    t = t[:t_len:t_downsample_ratio]
    
    plt.plot(t, np.mean(uu[:config['NE'],:t_len:t_downsample_ratio],axis=0) )  #population avg (ext neuron, then inh)
    #plt.plot(t, np.mean(uu[config['NE']:,:],axis=0) , '--')
    plt.xlabel('Time (ms)')
    plt.ylabel('Pop avg firing rate\n(sp/ms)')
    
    # rank neurons by firing rate
    mean_u = np.mean(uu[:, int(uu.shape[1]/2):], axis = 1)
    mean_ff = np.mean(ff[:, int(uu.shape[1]/2):], axis = 1)    
    sorted_id = np.argsort(mean_u)     
    #samples = np.arange(500,len(mean_u),2300)
    #samples = np.arange(200,10000,3100)
    samples = np.arange(200,10000,100)
    
    #copied color palette directly from https://carto.com/carto-colors/ 
    #colors = ['#fbe6c5','#f5ba98','#ee8a82','#dc7176','#c8586c','#9c3f5d','#70284a']
    colors = ['#fbe6c5','#f5ba98','#dc7176','#9c3f5d','#70284a']*100
    colors.reverse()
    #colors=colors[::2]
    
    plt.subplot(2,3,2)    
    for i in range(len(samples)):    
        plt.plot(t, uu[sorted_id[samples[i]],:t_len:t_downsample_ratio], color=colors[i])  #invidual neuron
    plt.xlabel('Time (ms)')
    plt.ylabel('Firing rates (sp/ms)')
    
    plt.subplot(2,3,3)
    plt.hist(mean_u[:config['NE']],50)
    #plt.hist(mean_u[config['NE']:],50)
    plt.xlabel('Firing rate (sp/ms)')
    
    plt.subplot(2,3,4)
    plt.plot(t, np.mean(ff[:config['NE'],:t_len:t_downsample_ratio], axis=0) )  #invidual neuron
    #plt.plot(t, np.mean(ff[config['NE']:,:], axis=0) , '--')  #invidual neuron
    plt.xlabel('Time (ms)')
    plt.ylabel('Pop avg Fano factor')
    
    plt.subplot(2,3,5)
    for i in range(len(samples)):    
        plt.plot(t, ff[sorted_id[samples[i]],:t_len:t_downsample_ratio], color=colors[i])  #invidual neuron        
    plt.xlabel('Time (ms)')
    plt.ylabel('Fano factor')
    
    plt.subplot(2,3,6)
    plt.hist(mean_ff[:config['NE']],51)#,np.linspace(0,1,51))    
    #plt.hist(mean_ff[config['NE']:],np.linspace(0,1,51))    
    plt.xlabel('Fano factor')
    
    plt.tight_layout()
    
    # also plot spatio-temporal pattern
    num_neurons = 500
    spatial_dowsample_ratio = int(uu.shape[0]/num_neurons) 
    
    tmax = 50
    temporal_downsample_ratio = int(tmax/config['dt']/1000)
    
    # PLOT 2D state space
    plt.figure()
    if indx == 35:  #plot the trajectory after convergence
        # frequency ~ 20 Hz = 0.02 kHz => period = 50 ms ~ 2.5 simulation time        
        for i in range(len(samples)):    
            tmp_x = uu[sorted_id[samples[i]],-200:]
            tmp_y = ff[sorted_id[samples[i]],-200:]*tmp_x
            plt.plot(tmp_x, tmp_y, color=colors[i])  #invidual neuron
            #plt.plot(tmp_x[-1], tmp_y[-1], '*', color='k')
    else: #plot the transient trajectory
        for i in range(len(samples)):    
            tmp_x = uu[sorted_id[samples[i]],:t_len:t_downsample_ratio]
            tmp_y = ff[sorted_id[samples[i]],:t_len:t_downsample_ratio]*tmp_x
            plt.plot(tmp_x, tmp_y, color=colors[i])  #invidual neuron
            plt.plot(tmp_x[-1], tmp_y[-1], '*', color='gray')
    plt.xlabel('Mean firing rate $\mu$ (sp/ms)')
    plt.ylabel('Firing variability $\sigma^2$ (sp^2/ms)')
    plt.tight_layout()
    
    # PLOT spatio-temporal pattern
    plt.figure()
    extent = [ 0, tmax*0.02 , 1, num_neurons]
    plt.subplot(2,1,1)
    plt.imshow( uu[sorted_id[::spatial_dowsample_ratio] , :int(tmax/config['dt']):temporal_downsample_ratio], extent=extent, aspect='auto',  interpolation='none')
    #plt.xlabel('Time (s)')
    plt.colorbar()
    plt.xticks([])
    plt.ylabel('Neuron index (ranked)')
    plt.title('Mean firing rate (sp/ms)')
    
    plt.subplot(2,1,2)
    plt.imshow( ff[sorted_id[::spatial_dowsample_ratio] , :int(tmax/config['dt']):temporal_downsample_ratio], extent=extent, aspect='auto',cmap='plasma', interpolation='none')
    plt.xlabel('Time (s)')
    plt.ylabel('Neuron index (ranked)')
    plt.title('Fano factor')
    plt.colorbar()
    plt.tight_layout()



#path = './runs/pre2023_brunel_delay_05_examples_2023_oct_16/'
path = './runs/pre2023_brunel_delay_05_slice_fine_2023_oct_16/'
meta_dat = np.load(path+'meta_data.npz', allow_pickle=True)
uext_array = meta_dat['uext_array']
ie_ratio_array = meta_dat['ie_ratio_array']

plt.close('all')
plot_example(path, 15)
#plot_example(path, 17) #crit point 1; very slow convergence as expected
plot_example(path, 25)
#plot_example(path, 32) #crit point 2; oscillation slowly decay away
plot_example(path, 35)
# i = np.where(uext_array==20)[0][0]
# plt.close('all')
# for j in range(8):
#     #j = np.where(ie_ratio_array==jj)[0][0]
#     indx = np.ravel_multi_index((i,j),[len(uext_array), len(ie_ratio_array)])       
#     plot_example(path, indx)
    
#%% manually look into the dynamics

path = './runs/pre2023_small_network_with_corr_slice_fine_2023_nov_21/'
dat = load_data(path, 30)
plt.imshow(dat['mnn_mean'], aspect='auto')





#cut_off = 800
#u = dat['mnn_mean'][:,cut_off:]
#s = dat['mnn_std'][:,cut_off:]


############################################################
#%% Plot slice (small network, with correlation)

#path = './runs/pre2023_small_network_with_corr_slice_fine_2023_nov_21/' #warning: T_mnn=20 not be long enough for convergence; need 100
#path = './runs/pre2023_small_network_delay)with_corr_fine_2023_nov_22/'

dat = np.load(path+'post_analysis.npz')

ie_ratio = dat['ie_ratio']
uext = dat['uext']
mean_pop_avg = dat['mean_pop_avg']
ff_pop_avg = dat['ff_pop_avg']
corr_pop_avg = dat['corr_pop_avg']
mean_pop_std = dat['mean_pop_std']
ff_pop_std = dat['ff_pop_std']
corr_pop_std = dat['corr_pop_std']

osc_amp = dat['osc_amp']
osc_freq = dat['osc_freq']
osc_amp_ff = dat['osc_amp_ff']
mean_quartiles = dat['mean_quartiles']
ff_quartiles = dat['ff_quartiles']

crit_pts = [3.4, 6.4]

plt.close('all')
plt.figure() #plot a slice
plt.subplot(2,2,1)
#plt.errorbar(ie_ratio, mean_pop_avg[0,:], mean_pop_std[0,:])   
plt.fill_between(ie_ratio, mean_pop_avg[0,:]-mean_pop_std[0,:], mean_pop_avg[0,:]+mean_pop_std[0,:], alpha=0.3)
#plt.fill_between(ie_ratio, mean_quartiles[0,:,0], mean_quartiles[0,:,1], alpha=0.3)
plt.plot(ie_ratio, mean_pop_avg[0,:])

plt.ylabel('Pop. avg. firing rate (sp/ms)')

#for p in crit_pts:
#    plt.plot([p,p],[-0.1,0.5],'--', color='gray')
#plt.ylim([-0.05,0.5])

plt.subplot(2,2,2)
#plt.errorbar(ie_ratio, ff_pop_avg[0,:], ff_pop_std[0,:])   
plt.fill_between(ie_ratio, ff_pop_avg[0,:]-ff_pop_std[0,:], ff_pop_avg[0,:]+ff_pop_std[0,:], alpha=0.3) 
#plt.fill_between(ie_ratio, ff_quartiles[0,:,0], ff_quartiles[0,:,1], alpha=0.3) 
plt.plot(ie_ratio, ff_pop_avg[0,:])   
plt.ylabel('Pop. avg. Fano factor')

#for p in crit_pts:
#    plt.plot([p,p],[-0.2,0.8],'--', color='gray')
#plt.ylim([-0.2,0.8])



plt.subplot(2,2,3)    
plt.plot(ie_ratio, osc_amp[0,:])
plt.ylabel('Oscillation amplitude (sp/ms)')
plt.xlabel('Inh-to-ext ratio')

# for p in crit_pts:
#     plt.plot([p,p],[-0.001,0.008],'--', color='gray')
# plt.ylim([-0.001,0.008])

plt.subplot(2,2,4)    
plt.plot(ie_ratio, osc_freq[0,:])
plt.ylabel('Oscillation frequency (Hz)')
plt.xlabel('Inh-to-ext ratio')

plt.tight_layout()    

# for p in crit_pts:
#     plt.plot([p,p],[-1,30],'--', color='gray')
# plt.ylim([-1,30])

plt.figure()
plt.subplot(2,2,1)
plt.fill_between(ie_ratio, corr_pop_avg[0,:]-corr_pop_std[0,:], corr_pop_avg[0,:]+corr_pop_std[0,:], alpha=0.3) 
plt.plot(ie_ratio,corr_pop_avg[0,:])
plt.ylabel('Avg. correlation')
plt.xlabel('Inh-to-ext ratio')

#%% small network, with correlation, and with delay, separate stats for E/I pops

path = './runs/pre2023_small_network_delay)with_corr_fine_2023_nov_22/'

dat = np.load(path+'post_analysis.npz')

ie_ratio = dat['ie_ratio']
uext = dat['uext']
mean_pop_avg = dat['mean_pop_avg'][:,:,0]
ff_pop_avg = dat['ff_pop_avg'][:,:,0]
corr_pop_avg = dat['corr_pop_avg'][:,:,0]
mean_pop_std = dat['mean_pop_std'][:,:,0]
ff_pop_std = dat['ff_pop_std'][:,:,0]
corr_pop_std = dat['corr_pop_std'][:,:,0]

osc_amp = dat['osc_amp']
osc_freq = dat['osc_freq']
osc_amp_ff = dat['osc_amp_ff']
mean_quartiles = dat['mean_quartiles']
ff_quartiles = dat['ff_quartiles']

crit_pts = [3.4, 6.4]

plt.close('all')
plt.figure() #plot a slice
plt.subplot(2,2,1)
#plt.errorbar(ie_ratio, mean_pop_avg[0,:], mean_pop_std[0,:])   
plt.fill_between(ie_ratio, mean_pop_avg[0,:]-mean_pop_std[0,:], mean_pop_avg[0,:]+mean_pop_std[0,:], alpha=0.3)
#plt.fill_between(ie_ratio, mean_quartiles[0,:,0], mean_quartiles[0,:,1], alpha=0.3)
plt.plot(ie_ratio, mean_pop_avg[0,:])

plt.ylabel('Pop. avg. firing rate (sp/ms)')

#for p in crit_pts:
#    plt.plot([p,p],[-0.1,0.5],'--', color='gray')
#plt.ylim([-0.05,0.5])

plt.subplot(2,2,2)
#plt.errorbar(ie_ratio, ff_pop_avg[0,:], ff_pop_std[0,:])   
plt.fill_between(ie_ratio, ff_pop_avg[0,:]-ff_pop_std[0,:], ff_pop_avg[0,:]+ff_pop_std[0,:], alpha=0.3) 
#plt.fill_between(ie_ratio, ff_quartiles[0,:,0], ff_quartiles[0,:,1], alpha=0.3) 
plt.plot(ie_ratio, ff_pop_avg[0,:])   
plt.ylabel('Pop. avg. Fano factor')

#for p in crit_pts:
#    plt.plot([p,p],[-0.2,0.8],'--', color='gray')
#plt.ylim([-0.2,0.8])



plt.subplot(2,2,3)    
plt.plot(ie_ratio, osc_amp[0,:])
plt.ylabel('Oscillation amplitude (sp/ms)')
plt.xlabel('Inh-to-ext ratio')

# for p in crit_pts:
#     plt.plot([p,p],[-0.001,0.008],'--', color='gray')
# plt.ylim([-0.001,0.008])

plt.subplot(2,2,4)    
plt.plot(ie_ratio, osc_freq[0,:])
plt.ylabel('Oscillation frequency (Hz)')
plt.xlabel('Inh-to-ext ratio')

plt.tight_layout()    

# for p in crit_pts:
#     plt.plot([p,p],[-1,30],'--', color='gray')
# plt.ylim([-1,30])

plt.figure()
plt.subplot(2,2,1)
plt.fill_between(ie_ratio, corr_pop_avg[0,:]-corr_pop_std[0,:], corr_pop_avg[0,:]+corr_pop_std[0,:], alpha=0.3) 
plt.plot(ie_ratio,corr_pop_avg[0,:])
plt.ylabel('Avg. correlation')
plt.xlabel('Inh-to-ext ratio')

    