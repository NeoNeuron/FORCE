# FORCE_external_feedback_loop_multi-task_test.py
#	
# Test the trained multi-task network
#
# MATLAB vertion written by David Sussillo
# Modified by Kai Chen
#
# Benchmarks:	
# 	10 pattern case
# 	pure numpy version: 	196.64 s
#	PyTorch CUDA version: 	 17.60 s
#	PyTorch CPU version: 	 29.19 s


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import torch
import json

def scan_corrcoef(x:torch.Tensor, y:torch.Tensor, arg:bool=False)->float:
	# define corrcoef for cuda
	def corrcoef(x:torch.Tensor, y:torch.Tensor)->torch.Tensor:
		return ((x*y).mean() - x.mean()*y.mean())/x.std()/y.std()
	# scan across time delay
	buffer = np.zeros(int(120/dt))
	for i in np.arange(1, len(buffer)+1):
		buffer[i-1] = corrcoef(x[i:], y[:-i]).cpu()	
	val_max = buffer[~np.isnan(buffer)].max()
	if arg:
		arg_max = np.argwhere(buffer==val_max)[0,0]
		return val_max, arg_max
	else:
		return val_max

if torch.cuda.is_available(): 	   # device agnostic code
    device = torch.device('cuda')  # and modularity
else:                              #
    device = torch.device('cpu')   #

mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['font.size'] = 16

# load parameters
with open('memory-task_3_cfg.json', 'r') as read_file:
	pm = json.load(read_file)
n_targets = pm['n_targets']
fname = f'memory-task_{n_targets:d}_net_hyper_pm.npz'
params = np.load(fname)
M=params['Jgg']
J_GI=params['Jgi']
wf = params['W_fb']
input_flag = np.nonzero(np.diff(params['I'])>0)[0]
input_flag = np.append(input_flag[1:], -1)

training_dym_data = np.load(f'memory-task_{n_targets:d}_training_dynamics.npz')

N = pm['N'] 
n_input = pm['n_input']
dt  = pm['dt']
tau = pm['tau']
n_rec2out = pm['n_rec2out']
n_test = 20

# create tensor in CUDA
M = torch.Tensor(M).to(device)
J_GI = torch.Tensor(J_GI).to(device)
# wo = torch.Tensor(wo).to(device)
wf = torch.Tensor(wf).to(device)

# print simulation setting
print('\tN: %d' % N)
print('\tn_rec2out: %d'% n_rec2out)

torch.cuda.synchronize()
t0 = time.time()
wo = training_dym_data['wt'][-1,:]
# weight manipulation
# %%
from sklearn.decomposition import PCA
# ft=training_dym_data['ft']
# wt=training_dym_data['wt']
# zt=training_dym_data['zt']
xt=training_dym_data['xt']
# %%
pca = PCA(n_components=N)
pca.fit(xt)

w_reduce = pca.transform(wo.reshape((1,-1)))
# print(w_reduce[0,:10])
# perturb one of the eigen-component of readout weight
w_reduce[0,200] = 0.0

wo = pca.inverse_transform(w_reduce)
wo = torch.Tensor(wo.flatten()).to(device)

# generate targets:
I_range = [1,5]
i_delay=1100 #ms
i_strengthes = np.linspace(I_range[0],I_range[1],50)
q_value = np.zeros_like(i_strengthes)

scale_factor = dt/tau
for idx, i_strength in enumerate(i_strengthes):
	input = np.ones(int(i_delay/dt))*i_strength

	input_bias = torch.Tensor(input.reshape((-1,1))).to(device)

	simtime = np.arange(0,input.shape[0])*dt
	simtime_len = len(simtime)

	torch.manual_seed(382)
	x0 = training_dym_data['xt'][-1,:]
	x0 = torch.Tensor(x0).to(device)
	z0 = 0.5*torch.zeros(1).to(device)

	xt = torch.zeros((simtime_len,N)).to(device)
	zt = torch.zeros(simtime_len).to(device)
	x = x0.clone()
	r = torch.tanh(x)
	z = z0.clone()
	for ti in np.arange(simtime_len):
		# sim, so x(t) and r(t) are created.
		x += torch.mul((-x + M @ r + wf*input_bias[ti]), scale_factor)
		r = torch.tanh(x)
		z = wo @ r

		# Store the output of the system.
		xt[ti,:]=x
		zt[ti]=z
	q_value[idx] = (0.5*torch.sum(torch.abs(input_bias.flatten()[-100:]-zt[-100:])**2)*torch.sum(wo**2)).cpu()
torch.cuda.synchronize()
print(f'evolve dynamics takes {time.time()-t0:.3f} s')

np.savez(f'memory-task_{n_targets:d}_qvalue_final_result.npz', q=np.array(q_value))
# %%
# import numpy as np
# import matplotlib.pyplot as plt
# data_package = np.load(f'memory-task_300_qvalue_result.npz')
# q_value=data_package['q']
# fname = f'memory-task_300_net_hyper_pm.npz'
# params = np.load(fname)
# input_flag = np.nonzero(np.diff(params['I'])>0)[0]
# input_value = params['I'][input_flag+1]
# n_test=20
# input_flag = np.append(input_flag[1:], -1)
# %%
# fig1, ax1 = plt.subplots(1,1,figsize=(8,5), sharex=True)
# I_range = [1,5]
# i_strengthes = np.linspace(I_range[0],I_range[1],20)
# trial_ids = np.arange(1,n_test+1)
# xx, yy = np.meshgrid(trial_ids, i_strengthes)
# pax = ax1.pcolormesh(xx, yy, np.log10(np.array(q_values).T), shading='nearest', cmap=plt.cm.summer, vmax=0, vmin=-4)
# plt.colorbar(pax, ax=ax1)
# ax1.plot(trial_ids, input_value[:n_test], '.k', ms=10)
# ax1.set_xlabel('Trials')
# ax1.set_xticks(trial_ids)
# ax1.set_yticks([1,2,3,4,5])
# ax1.set_ylabel('Value')
# ax1.set_title(f'q value')
# fig1.savefig(f'FORCE_Type_A_Memory_Task_300_qvalue.png')
# fig1.savefig(f'FORCE_Type_A_Memory_Task_{n_targets:d}_qvalue.png')
# save final frame as figure
fig2, ax2 = plt.subplots(1,1,figsize=(10,6))
ax2.semilogy(i_strengthes, q_value, color='navy')
ax2.set_title(f'Test')
ax2.set_xlabel('Target Value')
ax2.set_ylabel('q value')

plt.tight_layout()
fig2.savefig(f'FORCE_Type_A_Memory_Task_{n_targets:d}_qvalue_final.png')
# %%
