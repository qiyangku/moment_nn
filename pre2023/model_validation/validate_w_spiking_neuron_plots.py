# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 19:33:20 2021

@author: dell
"""

import numpy as np
import matplotlib.pyplot as plt
# import matplotlib as mpl
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Palatino"],
# })

def plot_maf():
    file = 'D:/MNN/moment_neural_network/runs/snn/validate_w_spk_neuron.npy'
    emp_u, emp_s, maf_u, maf_s, u, s = np.load(file, allow_pickle = True)
    
    
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)    
    
    color = ['tab:blue', 'tab:orange']
    label = [r'$\bar{\sigma} = 1$', r'$\bar{\sigma} = 2$']
    
    
    for j in range(len(s)):
        ax1.plot(u, maf_u[:,j], color = color[j], alpha = 0.8, label = label[j])
        ax1.plot(u, emp_u[:,j], '.', color = color[j])
    ax1.set_xlabel(r'Mean Input Current $\bar{\mu}$')
    ax1.set_ylabel(r'Mean Firing Rate $\mu$')
    
    
    ax2 = fig.add_subplot(1,2,2)    
    for j in range(len(s)):
        ax2.plot(u, maf_s[:,j], color = color[j], alpha = 0.8, label = label[j])
        ax2.plot(u, emp_s[:,j], '.', color = color[j])
    ax2.set_xlabel(r'Mean Input Current $\bar{\mu}$')
    ax2.set_ylabel(r'Firing Variability $\sigma$')
    
    ax1.legend()#['IF neuron','Moment Activation'])

def plot_corr(file_num):
    #file = './runs/snn/validate_corr_4.npy'
    file = 'D:/MNN/moment_neural_network/runs/snn/validate_corr_{}.npy'.format(file_num)
    file_quad = 'D:/MNN/moment_neural_network/runs/snn/validate_corr_{}_quad.npy'.format(file_num)
    dat = np.load(file, allow_pickle = True).item(0)
    dat_quad = np.load(file_quad, allow_pickle = True).item(0)
    
    fig = plt.figure()
    
    n = 0
    M, N = len(dat['u']), len(dat['s'])
    
    #plot all runs    
    for i in range(M):
        for j in range(N):
            n += 1
            ax = fig.add_subplot(M,N,n)
            ax.plot(dat['rho_in'], dat['rho_out'][:,i,j], '.')
            ax.plot(dat['rho_in'], dat['rho_maf'][:,i,j])
            ax.axis('equal')
            ax.set_xlim(-1,1)
            ax.set_ylim(-1,1)
            ax.set_xticks([])
            ax.set_yticks([])
            label_mu = r'$\bar{\mu}=$'+'{:.1f}'.format( dat['u'][i] )
            label_sig = r'$\bar{\sigma}=$'+'{:.1f}'.format(dat['s'][j])
            if i == 0:
                ax.set_title(label_sig)
            if j == 0:
                ax.set_ylabel(label_mu)
    
    #plot representative runs
    if file == 'D:/MNN/moment_neural_network/runs/snn/validate_corr_4.npy':        
        fig_rep = plt.figure()
        n = 0
        for i in range(4):
            for j in range(4):
                n += 1
                ax2 = fig_rep.add_subplot(4,4,n)
                ax2.plot(dat['rho_in'], dat['rho_out'][:,i,j+1], '.')
                ax2.plot(dat['rho_in'], dat['rho_maf'][:,i,j+1])
                ax2.plot(dat_quad['rho_in'], dat_quad['rho_maf'][:,i,j+1])
                ax2.axis('equal')
                ax2.set_xlim(-1,1)
                ax2.set_ylim(-1,1)
                if i < 3:
                    ax2.set_xticks([])
                if j > 0:
                    ax2.set_yticks([])
                label_mu = r'$\bar{\mu}=$'+str( np.round(dat['u'][i],1)  )
                label_sig = r'$\bar{\sigma}=$'+str(dat['s'][j+1])
                if i == 0:
                    ax2.set_title(label_sig)
                if j == 0:
                    ax2.set_ylabel(label_mu)
    elif file == 'D:/MNN/moment_neural_network/runs/snn/validate_corr_5.npy':        
        fig_rep = plt.figure()
        n = 0
        for i in range(4):
            for j in range(4):
                n += 1
                ax2 = fig_rep.add_subplot(4,4,n)
                ax2.plot(dat['rho_in'], dat['rho_out'][:,i,j*2+3], '.')
                ax2.plot(dat['rho_in'], dat['rho_maf'][:,i,j*2+3])
                ax2.plot(dat_quad['rho_in'], dat_quad['rho_maf'][:,i,j*2+3])
                ax2.axis('equal')
                ax2.set_xlim(-1,1)
                ax2.set_ylim(-1,1)
                if i < 3:
                    ax2.set_xticks([])
                if j > 0:
                    ax2.set_yticks([])
                label_mu = r'$\bar{\mu}=$'+'{:.1f}'.format( dat['u'][i] )
                label_sig = r'$\bar{\sigma}=$'+'{:.1f}'.format(dat['s'][j*2+3])
                if i == 0:
                    ax2.set_title(label_sig)
                if j == 0:
                    ax2.set_ylabel(label_mu)
            
if __name__=='__main__':
    #plot_maf()
    plot_corr(4)
#runfile('./apps/snn/validate_w_spiking_neuron_plots.py', wdir='./')