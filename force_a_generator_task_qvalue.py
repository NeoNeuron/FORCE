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
from scipy.interpolate import interp1d

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
with open('generator-task_20_cfg.json', 'r') as read_file:
	pm = json.load(read_file)
n_targets = pm['n_targets']
fname = f'generator-task_{n_targets:d}_net_hyper_pm.npz'
params = np.load(fname)
M=params['Jgg']
J_GI=params['Jgi']
wf = params['wf']
input_flag = np.nonzero(np.diff(params['I'])>0)[0]
input_flag = np.append(input_flag[1:], -1)

training_dym_data = np.load(f'generator-task_{n_targets:d}_training_dynamics.npz')

N = pm['N'] 
n_input = pm['n_input']
dt = pm['dt']
tau = pm['tau']
n_rec2out = pm['n_rec2out']
n_test = 10

# create tensor in CUDA
M = torch.Tensor(M).to(device)
J_GI = torch.Tensor(J_GI).to(device)
# wo = torch.Tensor(wo).to(device)
wf = torch.Tensor(wf).to(device)

# print simulation setting
print('\tN: %d' % N)
print('\tn_rec2out: %d'% n_rec2out)

q_values = []
torch.cuda.synchronize()
t0 = time.time()
for iidx in range(len(input_flag[:n_test])):
	wo = torch.Tensor(training_dym_data['wt'][input_flag[iidx],:]).to(device)
	# generate targets:
	I_range = [0.2,1.1] # Hz
	i_delay=500 #ms
	i_strengthes = np.linspace(I_range[0],I_range[1],10)
	q_value = np.zeros_like(i_strengthes)

	for idx, i_strength in enumerate(i_strengthes):
		period_len = int((2*np.pi/i_strength*1000)/dt)
		input = np.sin(np.arange(2*period_len)*dt*i_strength/1000)
		input = np.hstack((np.ones(int(i_delay/dt))*i_strength, input))

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
			x += dt/tau * (-x + M @ r + wf*input_bias[ti])
			r = torch.tanh(x)
			z = wo @ r

			# Store the output of the system.
			xt[ti,:]=x
			zt[ti]=z
		q_value[idx] = (0.5*torch.sum(torch.abs(input_bias.flatten()[-period_len:]-zt[-period_len:])**2)*torch.sum(wo**2)).cpu()
	q_values.append(q_value)
torch.cuda.synchronize()
print(f'evolve dynamics takes {time.time()-t0:.3f} s')

np.savez(f'generator-task_{n_targets:d}_qvalue_result.npz', ft = input, zts = zt.cpu(), xt=xt.cpu(), q=np.array(q_values))
# %%
# import numpy as np
# import matplotlib.pyplot as plt
# data_package = np.load(f'generator-task_20_qvalue_result.npz')
# q_values=data_package['q']
# fname = f'generator-task_20_net_hyper_pm.npz'
# params = np.load(fname)
input_flag = np.nonzero(np.diff(params['I'])>0)[0]
input_value = params['I'][input_flag+1]
# n_test=2
# %%
fig1, ax1 = plt.subplots(1,1,figsize=(6,5), sharex=True)
I_range = [0.2,1.1] # Hz
i_strengthes = np.linspace(I_range[0],I_range[1],10)
f = interp1d(i_strengthes, np.log10(q_values), axis=1)
i_strengthes_new = np.linspace(I_range[0],I_range[1],200)
trial_ids = np.arange(1,n_test+1)
xx, yy = np.meshgrid(trial_ids, i_strengthes_new)
pax = ax1.pcolormesh(xx, yy, f(i_strengthes_new).T, shading='nearest', cmap=plt.cm.summer,)
plt.colorbar(pax, ax=ax1)
ax1.plot(trial_ids, input_value[:n_test], '.k', ms=10)
ax1.set_xlabel('Trials')
ax1.set_xticks(trial_ids)
ax1.set_ylabel('Value (Hz)')
ax1.set_title(f'q_value')
plt.tight_layout()
fig1.savefig(f'FORCE_Type_A_Generator_Task_{n_targets:d}_qvalue.png')
# save final frame as figure
fig2, ax2 = plt.subplots(1,1,figsize=(20,5), sharex=True)
ax2.plot(simtime, input, color='red', label='target')
ax2.plot(simtime, zt.cpu(), color='orange', label='z', ls='--')
ax2.set_title(f'Test')
ax2.set_ylabel(r'$f$ and $z$')
ax2.legend()
ax2.set_xlabel('Time(ms)')


fig2.savefig(f'FORCE_Type_A_Generator_Task_{n_targets:d}_qvalue_trace.png')
# %%
