import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def gen_synaptic_weight(config):
    Ne = config['NE']
    Ni = config['NI']
    N = Ne+Ni
        
    if config['randseed'] is None:
        W = np.random.randn(N,N)
        coin_toss = np.random.rand(N,N)    
    else:
        rng = np.random.default_rng( config['randseed'] )
        W = rng.standard_normal(size=(N, N))
        coin_toss = rng.uniform(size=(N,N))
    
    #   excitatory weight
    W[:Ne,:Ne] = W[:Ne,:Ne]*config['wee']['std'] + config['wee']['mean']
    W[Ne:,:Ne] = W[Ne:,:Ne]*config['wie']['std'] + config['wie']['mean']    
    W[:,:Ne] = np.abs(W[:,:Ne])
    
    #   inhibitory weight
    W[:Ne,Ne:] = W[:Ne,Ne:]*config['wei']['std'] + config['wei']['mean']
    W[Ne:,Ne:] = W[Ne:,Ne:]*config['wii']['std'] + config['wii']['mean']
    W[:,Ne:] = -np.abs(W[:,Ne:])
    
    #apply connection probability (indegree should then be poisson)
    W[ coin_toss > config['conn_prob'] ] = 0
    
    #apply scaling
    W[:Ne,:Ne] *= 1/(Ne-1)
    W[Ne:,:Ne] *= 1/Ne
    W[:Ne,Ne:] *= 1/Ni    
    W[Ne:,Ne:] *= 1/(Ni-1)
    
    #remove diagonal (self-conneciton)
    np.fill_diagonal(W,0)
    
    
    if config['sparse_weight']:
        W = sp.sparse.csr_matrix(W) # W.dot() is efficient but not ().dot(W)        
    return W

class InputGenerator():
    def __init__(self, config):
        self.NE = config['NE']
        self.NI = config['NI']
        self.N = config['NE']+config['NI']
        self.var_ind = config['var_ind']        
        self.var_shr = config['var_shr']        
        
        #define external input mean
        self.input_mean = lambda s: s*np.ones((self.N,1)) 
        
        #calculate external input cov
        self.input_cov = self.var_ind*np.eye(self.N) + self.var_shr #diagonal = independent noise, otherwise shared
        #(see Moreno-Bote (2014), Supp Info Eq. 23)
        
        self.L_ext = np.linalg.cholesky(self.input_cov)        
        
        # assign coordinate to [0,2pi)
        self.theta_E = np.linspace(0, 2*np.pi, config['NE']+1)[:-1] 
        self.theta_I = np.linspace(0, 2*np.pi, config['NI']+1)[:-1] 
        
        return
    
    def gen_gaussian_current(self, s, dt):     
        u = self.input_mean(s)
        z = np.matmul(self.L_ext, np.random.randn(self.N,1))        
        I_ext = u*dt + z*np.sqrt(dt)
        return I_ext
    
    def update(self, mean_type = 'linear', cov_type = 'uniform'):
        ''' define different input functions'''
        if mean_type == 'linear':
            # default setting no change needed
            pass
        elif mean_type == 'gaussian':
            # replace inputs with gaussian tuning function
            print('method not implemented yet')
            pass
        
        if cov_type == 'uniform':
            # default setting, no change needed
            pass
        elif cov_type == 'cosine':
            #sec cov to sinusoidal function
            self.input_cov = self.cosine_cov()
            
            pass
        
    def cosine_cov(self):
        Cind = self.var_ind*np.eye(self.N)
        
        Cshr = np.zeros((self.N,self.N))
        k = 1 #spatial frequency
        x = self.var_shr*np.cos( k*self.theta_E)
        for i in range(self.NE): #only set excitatory input to be correlated
            Cshr[i,:self.NE] = np.roll(x,i)
        C = Cind + Cshr
        
        try:
            sp.linalg.cholesky(C)
        except:
            print('Warning: covariance matrix is not PSD!')
        
        return C
        
        
        

if __name__=='__main__':
    
    # test code
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
    
    config = gen_config(shared_noise_scale = 0.5)
    gen = InputGenerator(config)
    gen.update(cov_type = 'cosine')
    
    plt.close('all')
    plt.imshow(gen.input_cov)
    
        
        