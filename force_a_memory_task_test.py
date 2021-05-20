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
with open('memory-task_10_cfg.json', 'r') as read_file:
	pm = json.load(read_file)
n_targets = pm['n_targets']
fname = f'memory-task_{n_targets:d}_net_hyper_pm.npz'
params = np.load(fname)
M=params['Jgg']
J_GI=params['Jgi']
input_bias_set = np.array(params['I'])
wf = params['W_fb']

training_dym_data = np.load(f'memory-task_{n_targets:d}_training_dynamics.npz')
wo = training_dym_data['wt'][-1,:]

N = pm['N'] 
n_input = pm['n_input']
dt = pm['dt']
tau = pm['tau']
n_rec2out = pm['n_rec2out']
n_test = 20

# create tensor in CUDA
M = torch.Tensor(M).to(device)
J_GI = torch.Tensor(J_GI).to(device)
wo = torch.Tensor(wo).to(device)
wf = torch.Tensor(wf).to(device)

# print simulation setting
print('\tN: %d' % N)
print('\tn_rec2out: %d'% n_rec2out)

# generate targets:
I_range = [1,5]
I_duration=500 #ms
I_delay_range=[500, 6000] #ms
input = np.array([0])
target = np.array([0])
for i_strength, i_delay in zip(
	np.random.rand(n_test)*(I_range[1]-I_range[0])+I_range[0],
	np.random.rand(n_test)*(I_delay_range[1]-I_delay_range[0])+I_delay_range[0],
):
	input = np.append(input, np.ones(int(I_duration/dt))*i_strength)
	input = np.append(input, np.zeros(int(i_delay/dt)))
	target = np.append(target, np.ones(int((I_duration+i_delay)/dt))*i_strength)


input_bias = torch.Tensor(input.reshape((-1,1))).to(device)
ft = torch.Tensor(target).to(device)

simtime = np.arange(0,ft.shape[0])*dt
simtime_len = len(simtime)

torch.manual_seed(382)
x0 = training_dym_data['xt'][-1,:]
x0 = torch.Tensor(x0).to(device)
z0 = 0.5*torch.zeros(1).to(device)

zt_total = []
indices = []
torch.cuda.synchronize()
t0 = time.time()
zt = torch.zeros(simtime_len).to(device)
x = x0.clone()
r = torch.tanh(x)
z = z0.clone()
for ti in np.arange(simtime_len):
	# sim, so x(t) and r(t) are created.
	x += dt/tau * (-x + M @ r + wf*z + J_GI@input_bias[ti])
	r = torch.tanh(x)
	z = wo @ r

	# Store the output of the system.
	zt[ti]=z
# zt_total.append(np.array(zt.cpu()))
# _, idx = scan_corrcoef(ft, zt, arg=True)
# indices.append(idx)
error_avg = torch.sum(torch.abs(ft-zt))/(simtime_len)
print(f'Testing MAE:  {error_avg.cpu():3f}')

torch.cuda.synchronize()
print(f'evolve dynamics takes {time.time()-t0:.3f} s')

ft_cpu = np.array(ft.cpu())

np.savez(f'memory-task_{n_targets:d}_test_result.npz', ft = ft_cpu, zts = zt.cpu())

# save final frame as figure
fig2, ax2 = plt.subplots(1,1,figsize=(20,5), sharex=True)
ax2.plot(simtime, input, color='red', label='input')
ax2.plot(simtime, ft.cpu(), color='navy', label='f')
ax2.plot(simtime, zt.cpu(), color='orange', label='z', ls='--')
ax2.set_title(f'Test')
ax2.set_ylabel(r'$f$ and $z$')
ax2.legend()
ax2.set_xlabel('Time(ms)')

plt.tight_layout()

plt.savefig(f'FORCE_Type_A_Memory_Task_{n_targets:d}_Testing.png')