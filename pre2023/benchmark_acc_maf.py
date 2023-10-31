from pre2023.model_validation.validate_w_spiking_neuron import input_output_anlaysis,InteNFire


emp_u, emp_s, maf_u, maf_s, u, s = input_output_anlaysis(input_type = 'spike')

#maybe I should increase the # of neurons (# trials), and decrease simulation time not 10s but 2s is enougth?
#so better parallelization and speed
#omg this snn simulation takes forever, thanks to dt=0.001 ms. So 10^6 steps needed per 1 s simulation

np.savez('benchmark_acc_maf.npz', emp_u=emp_u, emp_s=emp_s, maf_u=maf_u, maf_s=maf_s, u=u, s=s)

