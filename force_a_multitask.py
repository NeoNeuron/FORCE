# FORCE.py
#
# This function generates the sum of 4 sine waves in figure 2D using the architecture of figure 1A with the RLS
# learning rule.
#
# MATLAB vertion written by David Sussillo
# Modified by Kai Chen
#
# Benchmarks:	
# 	2 pattern case
#		pure numpy version: 	96.001 s
#		PyTorch CUDA version: 	16.001 s
#		PyTorch CPU version: 	31.121 s
#	10 pattern case
#		PyTorch CUDA version: 	59.001 s


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import torch
import json

if torch.cuda.is_available(): 	   # device agnostic code
    device = torch.device('cuda')  # and modularity
else:                              #
    device = torch.device('cpu')   #

mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['font.size'] = 14
mpl.rcParams['font.weight'] = 'bold'

param = {}
torch.cuda.synchronize()
t0 = time.time()
N = param['N'] = 1000
n_input = param['n_input'] = 100
p = param['p'] = 0.1
g = param['g'] = 1.5		# g greater than 1 leads to chaotic networks.
alpha = param['alpha'] = 0.0125
nsecs_train = param['nsecs_train'] = 3400
dt = param['dt'] = 0.1
learn_every = param['learn_every'] = 2

# sparse matrix M
M = np.random.rand(N,N)
M = (M<p).astype(int)
scale = 1.0/np.sqrt(p*N)
M = M * g*scale * np.random.randn(N,N)
M = torch.Tensor(M).to(device)

n_rec2out = param['n_rec2out'] = N
wo = torch.zeros(n_rec2out).to(device)
dw = torch.zeros(n_rec2out).to(device)
wf = 2.0*(torch.rand(N)-0.5).to(device)

# generate connectivity matrix for control inputs
J_GI = np.zeros((N, n_input))
col_indices = np.random.randint(n_input, size=N)
J_GI[np.arange(N, dtype=int), col_indices] = np.random.randn(N)
J_GI = torch.Tensor(J_GI).to(device)

# print simulation setting
print('\tN: %d' % N)
print('\tg: %.3f' % g)
print('\tp: %.3f' % p)
print('\tn_rec2out: %d'% n_rec2out)
print('\talpha: %.3f' % alpha)
print('\tnsecs train: %d' % nsecs_train)
print('\tlearn_every: %d' % learn_every)

simtime = np.arange(0,nsecs_train,dt)
simtime_len = len(simtime)

from gen_patterns import gen_random, gen_sequential

n_targets = param['n_targets'] = 4
seed = param['seed'] = 0
# fts_train = gen_random(num=n_targets, seed=seed, time=simtime)
fts_train = gen_sequential(num=n_targets, time=simtime)

# shifting order again
# --------------------
shuffle_toggle = param['shuffle_toggle'] = False
shuffle_seed = param['shuffle_seed'] = 0
if shuffle_toggle:
	order = np.arange(n_targets, dtype=int)
	param['order'] = order
	rg = np.random.Generator(np.random.MT19937(shuffle_seed))
	rg.shuffle(order)
	fts_train = [fts_train[idx] for idx in order]

ft = np.hstack(fts_train)

input_bias_set = 1.6*(np.random.rand(n_targets, n_input)-0.5)
input_bias = np.repeat(input_bias_set, simtime_len, axis=0)

# convert to GPU memory
ft = torch.Tensor(ft).to(device)
input_bias = torch.Tensor(input_bias).to(device)

simtime_len = len(ft)
simtime = np.arange(simtime_len) * dt

# create container for buffer variables
wt = torch.zeros((simtime_len, n_rec2out)).to(device)
zt = torch.zeros(simtime_len).to(device)
x0 = 0.5*torch.randn(N).to(device)
z0 = 0.5*torch.randn(1).to(device)

x = x0.clone()
r = torch.tanh(x)
z = z0.clone()

P = (1.0/alpha)*torch.eye(n_rec2out).to(device)
torch.cuda.synchronize()
print(f'matrix init takes {time.time()-t0:.3f} s')

torch.cuda.synchronize()
t0 = time.time()
for ti in np.arange(simtime_len):
    # sim, so x(t) and r(t) are created.
	x += dt * (-x + M @ r + wf*z + J_GI@input_bias[ti])
	r = torch.tanh(x)
	z = wo @ r
    
	if ((ti+1) % learn_every) == 0:
		# update inverse correlation matrix
		k = P @ r
		rPr = r @ k
		c = 1.0/(1.0 + rPr)
		P -= c *torch.outer(k,k)
		
		# update the error for the linear readout
		e = z-ft[ti]
		
		# update the output weights
		dw = -e*k*c	
		wo += dw
    
    # Store the output of the system.
	zt[ti] = z
	wt[ti,:] = wo

torch.cuda.synchronize()
print(f'evolve dynamics takes {time.time()-t0:.3f} s')

# save trained model parameters
ft_cpu = np.array(ft.cpu(), dtype=float)
wt_cpu = np.array(wt.cpu(), dtype=float)
zt_cpu = np.array(zt.cpu(), dtype=float)
with open(f'multi-task_{n_targets:d}_cfg.json', 'w') as write_file:
    json.dump(param, write_file, indent=2)
np.savez(f'multi-task_{n_targets:d}_net_hyper_pm.npz', Jgg=M.cpu(), Jgi=J_GI.cpu(), I = input_bias_set, wf = wf.cpu())

np.savez(f'multi-task_{n_targets:d}_training_dynamics.npz', ft=ft_cpu, wt=wt_cpu, zt = zt_cpu)


# print training error
error_avg = torch.sum(torch.abs(zt-ft))/simtime_len
print(f'Training MAE:  {error_avg.cpu():3e}')    

# save final frame as figure
fig2, ax2 = plt.subplots(2,1,figsize=(20,10))
ax2[0].plot(simtime, ft_cpu, color='green', label='f')
ax2[0].plot(simtime, zt_cpu, color='red', label='z')
ax2[0].set_title('Training')
ax2[0].set_xlabel('Time')
ax2[0].set_ylabel(r'$f$ and $z$')
ax2[0].legend()

ax2[1].plot(simtime, np.sqrt(np.sum(wt_cpu**2,axis=1)), label='|w|')[0]
ax2[1].set_xlabel('Time')
ax2[1].set_ylabel(r'|$w$|')
ax2[1].legend()
plt.tight_layout()

plt.savefig('FORCE_Type_A_Multitask_Training.png')