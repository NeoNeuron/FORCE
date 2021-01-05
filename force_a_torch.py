# FORCE.py
#
# This function generates the sum of 4 sine waves in figure 2D using the architecture of figure 1A with the RLS
# learning rule.
#
# MATLAB vertion written by David Sussillo
# Modified by Kai Chen
# Benchmarks:
#	2000 training, 2000 testing.
#		PyTorch CUDA:	7.456 s
#		PyTorch CPU:	13.627 s

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
p = param['p'] = 0.1
g = param['g'] = 1.5		# g greater than 1 leads to chaotic networks.
alpha = param['alpha'] = 1.0
nsecs = param['nsecs'] = 4000
nsecs_train = param['nsecs_train'] = 2000
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

# print simulation setting
print('\tN: %d' % N)
print('\tg: %.3f' % g)
print('\tp: %.3f' % p)
print('\tn_rec2out: %d'% n_rec2out)
print('\talpha: %.3f' % alpha)
print('\tnsecs: %d' % nsecs)
print('\tnsecs for train: %d' % nsecs_train)
print('\tlearn_every: %d' % learn_every)

simtime = np.arange(0,nsecs,dt)
simtime_len = len(simtime)
simtime_train_len = int(simtime_len/nsecs*nsecs_train)

from gen_patterns import gen_target
amps = 1.3 / np.array([1.0, 2.0, 6.0, 3.0])
ft = torch.Tensor(gen_target(amps, simtime)).to(device)

wt = torch.zeros((simtime_len,n_rec2out)).to(device)
zt = torch.zeros(simtime_len).to(device)
x0 = 0.5*torch.randn(N).to(device)
z0 = 0.5*torch.randn(1).to(device)

x = x0 
r = torch.tanh(x)
z = z0

P = (1.0/alpha)*torch.eye(n_rec2out).to(device)
torch.cuda.synchronize()
print(f'matrix init takes {time.time()-t0:.3f} s')
torch.cuda.synchronize()
t0 = time.time()
for ti in np.arange(len(simtime)):
    # sim, so x(t) and r(t) are created.
	# delayed and nonlinear distortion
	# if ti >=10:
	# 	x += dt*(-x+M@r+wf*1.3*np.tanh(np.sin(np.pi*zt[ti-10])))
	# else:
	# 	x += dt * (-x + M @ r)
	x += dt * (-x + M @ r + wf*z)
	r = torch.tanh(x)
	z = wo @ r
    
	if ((ti+1) % learn_every) == 0 and ti < simtime_train_len:
		# update inverse correlation matrix
		k = P @ r
		rPr = r @ k
		c = 1.0/(1.0 + rPr)
		P -= c * torch.outer(k,k)
		
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
# save outputs
# ------------
ft_cpu = np.array(ft.cpu(), dtype=float)
wt_cpu = np.array(wt.cpu(), dtype=float)
zt_cpu = np.array(zt.cpu(), dtype=float)
with open(f'net_1_cfg.json', 'w') as write_file:
    json.dump(param, write_file, indent=2)
np.savez('net_1_training_dynamics.npz', ft=ft_cpu, wt=wt_cpu, zt = zt_cpu)

np.savez('net_1_hyper.npz', Jgg=M.cpu(), wf = wf.cpu())

# print training error
error_avg = torch.sum(torch.abs(zt[:simtime_train_len]-ft[:simtime_train_len]))/simtime_train_len
print(f'Training MAE:  {error_avg.cpu():3f}')    

error_avg = torch.sum(torch.abs(zt[simtime_train_len:]-ft[simtime_train_len:]))/simtime_train_len
print(f'Testing MAE:  {error_avg.cpu():3f}')

# save final frame as figure
fig2, ax2 = plt.subplots(2,1,figsize=(20,10), sharex=True)
ax2[0].plot(simtime, ft_cpu, color='green', label='f')
ax2[0].plot(simtime, zt_cpu, color='red', label='z')
ax2[0].set_title('Training')
ax2[0].set_xlabel('Time')
ax2[0].set_ylabel(r'$f$ and $z$')
ax2[0].axvline(simtime[simtime_train_len-1],color='cyan')
ax2[0].legend()

wo_len = np.sqrt(np.sum(wt_cpu**2,axis=1))
ax2[1].plot(simtime, wo_len, label='|w|')[0]
ax2[1].set_xlabel('Time')
ax2[1].set_ylabel(r'|$w$|')
ax2[1].set_xlim(0, nsecs)
ax2[1].axvline(simtime[simtime_train_len-1],color='cyan')
ax2[1].legend()
plt.tight_layout()

plt.savefig('Figure_net_1.png')