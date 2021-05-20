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
mpl.rcParams['font.size'] = 16

param = {}
torch.cuda.synchronize()
t0 = time.time()
N = param['N'] = 500
n_input = param['n_input'] = 1
p = param['p'] = 1
g = param['g'] = 1.2		# g greater than 1 leads to chaotic networks.
alpha = param['alpha'] = 10
nsecs_train = param['nsecs_train'] = 3400
dt = param['dt'] = 1
tau = param['tau'] = 100 #ms
learn_every = param['learn_every'] = 10

# sparse matrix M
M = np.random.rand(N,N)
M = (M<p).astype(int)
scale = g/np.sqrt(p*N)
M = M * scale * np.random.randn(N,N)
M = torch.Tensor(M).to(device)

n_rec2out = param['n_rec2out'] = N
wo = torch.zeros(n_rec2out).to(device)
dw = torch.zeros(n_rec2out).to(device)
wf = 2.0*(torch.rand(N)-0.5).to(device)

# generate connectivity matrix for control inputs
J_GI = 2.0*(torch.rand(N, 1)-0.5).to(device)

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

# generate targets:
n_targets = param['n_targets'] = 20
seed = param['seed'] = 0
np.random.seed(seed)
I_range = [0.2,1.1] # Hz
input = np.array([0])
target = np.array([0])
for i_strength in np.random.rand(n_targets)*(I_range[1]-I_range[0])+I_range[0]:
	i_duration= 4*np.pi/i_strength * 1000
	i_delay= 20*np.pi/i_strength * 1000
	input = np.append(input, np.ones(int(i_duration/dt))*i_strength)
	input = np.append(input, np.zeros(int(i_delay/dt)))
	target = np.append(target, np.zeros(int(i_duration/dt)))
	target = np.append(target, np.sin(np.arange(int(i_delay/dt))*dt*i_strength/1000))

# convert to GPU memory
input_bias = torch.Tensor(input.reshape((-1,1))).to(device)
ft = torch.Tensor(target).to(device)

simtime_len = len(ft)
simtime = np.arange(simtime_len) * dt

# create container for buffer variables
# wt = torch.zeros((simtime_len, n_rec2out)).to(device)
# xt = torch.zeros((simtime_len, N)).to(device)
# zt = torch.zeros(simtime_len).to(device)
wt = np.zeros((simtime_len, n_rec2out))
xt = np.zeros((simtime_len, N))
zt = np.zeros(simtime_len)
x = 0.5*torch.randn(N).to(device)
z = 0.5*torch.randn(1).to(device)

r = torch.tanh(x)

P = (1.0/alpha)*torch.eye(n_rec2out).to(device)
torch.cuda.synchronize()
print(f'matrix init takes {time.time()-t0:.3f} s')

torch.cuda.synchronize()
t0 = time.time()
for ti in np.arange(simtime_len):
    # sim, so x(t) and r(t) are created.
	x += dt/tau * (-x + M @ r + wf*z + J_GI@input_bias[ti])
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
	xt[ti,:] = x.cpu()
	zt[ti] = z.cpu()
	wt[ti,:] = wo.cpu()

torch.cuda.synchronize()
print(f'evolve dynamics takes {time.time()-t0:.3f} s')

# save trained model parameters
ft_cpu = np.array(ft.cpu(), dtype=float)
# wt_cpu = np.array(wt.cpu(), dtype=float)
# zt_cpu = np.array(zt.cpu(), dtype=float)
with open(f'generator-task_{n_targets:d}_cfg.json', 'w') as write_file:
    json.dump(param, write_file, indent=2)
np.savez(f'generator-task_{n_targets:d}_net_hyper_pm.npz', Jgg=M.cpu(), Jgi=J_GI.cpu(), I = input, wf = wf.cpu())

np.savez(f'generator-task_{n_targets:d}_training_dynamics.npz', ft=ft_cpu, wt=wt, zt = zt, xt=xt)


# print training error
error_avg = np.sum(np.abs(zt-target))/simtime_len
print(f'Training MAE:  {error_avg:3e}')    

# save final frame as figure
fig2, ax2 = plt.subplots(2,1,figsize=(20,10))
ax2[0].plot(simtime, input, color='red', label='input')
ax2[0].plot(simtime, ft_cpu, color='navy', label='f')
ax2[0].plot(simtime, zt, color='orange', label='z', ls='--')
ax2[0].set_title('Training')
ax2[0].set_xlabel('Time(ms)')
ax2[0].set_ylabel(r'$f$ and $z$')
ax2[0].legend()

ax2[1].plot(simtime, np.sqrt(np.sum(wt**2,axis=1)), label='|w|')[0]
ax2[1].set_xlabel('Time(ms)')
ax2[1].set_ylabel(r'|$w$|')
ax2[1].legend()
plt.tight_layout()

plt.savefig(f'FORCE_Type_A_Generator_Task_{n_targets:d}_Training.png')