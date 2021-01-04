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
from matplotlib.animation import FuncAnimation
import time
import torch

if torch.cuda.is_available(): 	   # device agnostic code
    device = torch.device('cuda')  # and modularity
else:                              #
    device = torch.device('cpu')   #

mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['font.size'] = 14
mpl.rcParams['font.weight'] = 'bold'

# load parameters
fname = 'trained_net_10.npz'
params = np.load(fname)
M=params['Jgg']
J_GI=params['Jgi']
input_bias_set = np.array(params['I'])
wo = params['w']
options = params['options']
wf = params['wf']

N = M.shape[0]
n_input = J_GI.shape[1]
dt = 0.1
nRec2Out = N
nsecs = 1200
M = torch.Tensor(M).to(device)
J_GI = torch.Tensor(J_GI).to(device)
wo = torch.Tensor(wo).to(device)
wf = torch.Tensor(wf).to(device)

# print simulation setting
print('\tN: %d' % N)
print('\tnRec2Out: %d'% nRec2Out)

simtime = np.arange(0,nsecs,dt)
simtime_len = len(simtime)

def gen_target(amps, freqs, time):
	ft = np.zeros_like(time)
	for amp, freq in zip(amps, freqs):
		ft += amp*np.sin(np.pi*freq*time)
	ft = ft/1.5
	return ft

amps = [1.3/np.array(item) for item in options]
freqs = 1/60 * np.array([1.0, 2.0, 3.0, 4.0])

fts = [torch.Tensor(gen_target(amp, freqs, simtime)).to(device) for amp in amps]
single_len = len(fts[0])

input_bias = np.repeat(input_bias_set.reshape((input_bias_set.shape[0], 1, input_bias_set.shape[1])), 
						simtime_len, axis=1)

input_bias = torch.Tensor(input_bias).to(device)

x0 = 0.5*torch.randn(N).to(device)
z0 = 0.5*torch.randn(1).to(device)

zt_total = []
torch.cuda.synchronize()
t0 = time.time()
for iter in range(len(input_bias_set)):
	zt = torch.zeros(simtime_len).to(device)
	x = x0 
	r = torch.tanh(x)
	z = z0
	for ti in np.arange(simtime_len):
		# sim, so x(t) and r(t) are created.
		x += dt * (-x + M @ r + wf*z + J_GI@input_bias[iter,ti])
		r = torch.tanh(x)
		z = wo @ r
	
		# Store the output of the system.
		zt[ti]=z
	zt_total.append(np.array(zt.cpu()))

torch.cuda.synchronize()
print(f'evolve dynamics takes {time.time()-t0:.3f} s')

fts_cpu = np.array([np.array(ft.cpu()) for ft in fts])

np.savez(fname.split('.')[0]+'_test_data.npz', fts = fts_cpu, zts = zt_total)

# save final frame as figure
fig2, ax2 = plt.subplots(len(input_bias_set),1,figsize=(10,20),sharex=True)
for idx, ax_i in enumerate(ax2):
	ax_i.plot(simtime, fts[idx].cpu(), color='green', label='f')
	ax_i.plot(simtime, zt_total[idx], color='red', label='z')
	ax_i.set_title(f'Test {idx:d}')
	ax_i.set_ylabel(r'$f$ and $z$')
	ax_i.legend()
ax2[-1].set_xlabel('Time')

plt.tight_layout()

plt.savefig('Figure3_test.png')