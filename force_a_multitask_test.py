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
mpl.rcParams['font.size'] = 14
mpl.rcParams['font.weight'] = 'bold'

# load parameters
with open('multi-task_4_cfg.json', 'r') as read_file:
	pm = json.load(read_file)
n_targets = pm['n_targets']
fname = f'multi-task_{n_targets:d}_net_hyper_pm.npz'
params = np.load(fname)
M=params['Jgg']
J_GI=params['Jgi']
input_bias_set = np.array(params['I'])
wf = params['wf']

training_dym_data = np.load(f'multi-task_{n_targets:d}_training_dynamics.npz')
wo = training_dym_data['wt'][-1,:]

N = pm['N'] 
n_input = pm['n_input']
dt = pm['dt']
n_rec2out = pm['n_rec2out']
nsecs = 2000

# create tensor in CUDA
M = torch.Tensor(M).to(device)
J_GI = torch.Tensor(J_GI).to(device)
wo = torch.Tensor(wo).to(device)
wf = torch.Tensor(wf).to(device)

# print simulation setting
print('\tN: %d' % N)
print('\tn_rec2out: %d'% n_rec2out)

simtime = np.arange(0,nsecs,dt)
simtime_len = len(simtime)

from gen_patterns import gen_random, gen_sequential
# fts = gen_random(num=n_targets, seed=pm['seed'], time = simtime)
fts = gen_sequential(num=n_targets, time = simtime)
single_len = len(fts[0])
fts = torch.Tensor(np.array(fts)).to(device)

input_bias = np.repeat(input_bias_set.reshape((input_bias_set.shape[0], 1, input_bias_set.shape[1])), 
						simtime_len, axis=1)

input_bias = torch.Tensor(input_bias).to(device)

torch.manual_seed(382)
x0 = 0.5*torch.randn(N).to(device)
z0 = 0.5*torch.randn(1).to(device)

zt_total = []
indices = []
torch.cuda.synchronize()
t0 = time.time()
for ft, input in zip(fts, input_bias):
	zt = torch.zeros(simtime_len).to(device)
	x = x0.clone()
	r = torch.tanh(x)
	z = z0.clone()
	for ti in np.arange(simtime_len):
		# sim, so x(t) and r(t) are created.
		x += dt * (-x + M @ r + wf*z + J_GI@input[ti])
		r = torch.tanh(x)
		z = wo @ r
	
		# Store the output of the system.
		zt[ti]=z
	zt_total.append(np.array(zt.cpu()))
	_, idx = scan_corrcoef(ft, zt, arg=True)
	indices.append(idx)
	error_avg = torch.sum(torch.abs(ft[idx:]-zt[:-idx]))/(simtime_len-idx)
	print(f'Testing MAE:  {error_avg.cpu():3f}')

torch.cuda.synchronize()
print(f'evolve dynamics takes {time.time()-t0:.3f} s')

fts_cpu = np.array(fts.cpu())

np.savez(f'multi-task_{n_targets:d}_test_result.npz', fts = fts_cpu, zts = zt_total)

# save final frame as figure
fig2, ax2 = plt.subplots(len(input_bias_set),1,figsize=(10,20), sharex=True)
for idx, ax_i in enumerate(ax2):
	ax_i.plot(simtime[indices[idx]+1:], fts[idx].cpu()[indices[idx]+1:], color='green', label='f')
	ax_i.plot(simtime[indices[idx]+1:], zt_total[idx][:-indices[idx]-1], color='red', label='z')
	ax_i.set_title(f'Test {idx+1:d}')
	ax_i.set_ylabel(r'$f$ and $z$')
	ax_i.legend()
ax2[-1].set_xlabel('Time')

plt.tight_layout()

plt.savefig('FORCE_Type_A_Multitask_Testing.png')