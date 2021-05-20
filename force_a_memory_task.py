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
#		# Torch CUDA version: 	16.001 s
#		PyTorch CPU version: 	31.121 s
#	10 pattern case
#		PyTorch CUDA version: 	59.001 s


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import torch
import json

def RLS_inplace(w, error, P, r):
	# update inverse correlation matrix
	k = P @ r
	# rPr = r @ k
	c = 1.0/(1.0 + r @ k)
	P -= c * (k.reshape((-1,1)) @ k.reshape((1,-1)))
	# update the output weights
	# dw = -error*k*c	
	# w += dw
	w -= error*k*c
	return w, P

def LMS_inplace(w, error, eta, r):
	# update the output weights
	w -= eta*error*r
	return w


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
dt = param['dt'] = 1
tau = param['tau'] = 100 #ms
learn_every = param['learn_every'] = 2

# sparse matrix M
M = np.random.rand(N,N)
M = (M<p).astype(int)
scale = g/np.sqrt(p*N)
M = M * scale * np.random.randn(N,N)
M = torch.Tensor(M).to(device)

n_rec2out = param['n_rec2out'] = N
W_out = torch.zeros(n_rec2out).to(device)
W_fb = 2.0*(torch.rand(N)-0.5).to(device)

# generate connectivity matrix for control inputs
J_GI = 2.0*(torch.rand(N, 1)-0.5).to(device)

# print simulation setting
print('\tN: %d' % N)
print('\tg: %.3f' % g)
print('\tp: %.3f' % p)
print('\tn_rec2out: %d'% n_rec2out)
print('\talpha: %.3f' % alpha)
print('\tlearn_every: %d' % learn_every)

# generate targets:
n_targets = param['n_targets'] = 10
seed = param['seed'] = 10
np.random.seed(seed)
I_range = [1,5]
I_duration=500 #ms
I_delay_range=[500, 6000] #ms
input = np.array([0])
target = np.array([0])
training_flag = np.array([False], dtype=bool)
for i_strength, i_delay in zip(
	# np.random.rand(n_targets)*(I_range[1]-I_range[0])+I_range[0],
	(1.8, 3.0, 4.9, 3.8, 2.6, 4.6, 1.4, 1.05, 2.25, 4.2),
	np.random.rand(n_targets)*(I_delay_range[1]-I_delay_range[0])+I_delay_range[0],
):
	input = np.append(input, np.ones(int(I_duration/dt))*i_strength)
	input = np.append(input, np.zeros(int(i_delay/dt)))
	target = np.append(target, np.ones(int((I_duration+i_delay)/dt))*i_strength)
	training_flag = np.append(training_flag, np.zeros(int(I_duration/dt), dtype=bool))
	training_flag = np.append(training_flag, np.ones(int(i_delay/dt), dtype=bool))

# convert to GPU memory
input_bias = torch.Tensor(input.reshape((-1,1))).to(device)
ft = torch.Tensor(target).to(device)

simtime_len = len(ft)
simtime = np.arange(simtime_len) * dt

# create container for buffer variables
# wt = np.zeros((simtime_len, n_rec2out))
# xt = np.zeros((simtime_len, N))
# zt = np.zeros(simtime_len)
wt = np.zeros((1, n_rec2out))
xt = np.zeros((1, N))
zt = np.zeros(1)
x = 0.5*torch.randn(N).to(device)
x = 0.5*torch.randn(N).to(device)
z = 0.5*torch.randn(1).to(device)

r = torch.tanh(x)

P = (1.0/alpha)*torch.eye(n_rec2out).to(device)
torch.cuda.synchronize()
print(f'matrix init takes {time.time()-t0:.3f} s')

buffer_len = int(simtime_len/10)
wt_gpu_buffer = torch.zeros((buffer_len, n_rec2out)).to(device)
xt_gpu_buffer = torch.zeros((buffer_len, N)).to(device)
zt_gpu_buffer = torch.zeros(buffer_len).to(device)
torch.cuda.synchronize()
t0 = time.time()
buffer_counter = 0
for ti in np.arange(simtime_len):
    # sim, so x(t) and r(t) are created.
	x += dt/tau * (-x + M @ r + W_fb*z + J_GI@input_bias[ti])
	r = torch.tanh(x)
	z = W_out @ r
    
	if ((ti+1) % learn_every) == 0: # and training_flag[ti]:
		P = (1.0/alpha)*torch.eye(n_rec2out).to(device)
		# update the error for the linear readout
		e = z-ft[ti]
		# inplace update
		RLS_inplace(w=W_out, error=e, P=P, r=r)
		# LMS_inplace(w=W_out, error=e, eta=1e-3, r=r)

    # Store the output of the system.
	xt_gpu_buffer[buffer_counter,:] = x
	wt_gpu_buffer[buffer_counter,:] = W_out
	zt_gpu_buffer[buffer_counter] = z
	buffer_counter+=1
	if buffer_counter == buffer_len:
		xt = np.append(xt, xt_gpu_buffer.cpu(), axis=0)
		wt = np.append(wt, wt_gpu_buffer.cpu(), axis=0)
		zt = np.append(zt, zt_gpu_buffer.cpu(), axis=0)
		buffer_counter=0
xt = np.append(xt, xt_gpu_buffer[:buffer_counter,:].cpu(), axis=0)
wt = np.append(wt, wt_gpu_buffer[:buffer_counter,:].cpu(), axis=0)
zt = np.append(zt, zt_gpu_buffer[:buffer_counter].cpu(), axis=0)
		

torch.cuda.synchronize()
print(f'evolve dynamics takes {time.time()-t0:.3f} s')

# save trained model parameters
ft_cpu = np.array(ft.cpu(), dtype=float)
wt = wt[1:,:]
xt = xt[1:,:]
zt = zt[1:]
# wt = np.array(wt.cpu(), dtype=float)
# zt = np.array(zt.cpu(), dtype=float)
# xt = np.array(xt.cpu(), dtype=float)
with open(f'memory-task_{n_targets:d}_cfg.json', 'w') as write_file:
    json.dump(param, write_file, indent=2)
np.savez(f'memory-task_{n_targets:d}_net_hyper_pm.npz', Jgg=M.cpu(), Jgi=J_GI.cpu(), I = input, W_fb = W_fb.cpu())

np.savez(f'memory-task_{n_targets:d}_training_dynamics.npz', ft=ft_cpu, wt=wt, zt = zt, xt=xt)


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

plt.savefig(f'FORCE_Type_A_Memory_Task_{n_targets:d}_Training.png')