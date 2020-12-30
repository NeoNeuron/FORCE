# FORCE.py
#
# This function generates the sum of 4 sine waves in figure 2D using the architecture of figure 1A with the RLS
# learning rule.
#
# MATLAB vertion written by David Sussillo
# Modified by Kai Chen

import numpy as np
from scipy.sparse import csr_matrix
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

torch.cuda.synchronize()
t0 = time.time()
N = 1000
p = 0.1
g = 1.5		# g greater than 1 leads to chaotic networks.
alpha = 1.0
nsecs = 4000
nsecs_train = 2000
dt = 0.1
learn_every = 2

# sparse matrix M
M = np.random.rand(N,N)
M = (M<p).astype(int)
scale = 1.0/np.sqrt(p*N)
M = M * g*scale * np.random.randn(N,N)
M = torch.Tensor(M).to(device)
# M_spa = csr_matrix(M)

nRec2Out = N
wo = torch.zeros(nRec2Out).to(device)
dw = torch.zeros(nRec2Out).to(device)
wf = 2.0*(torch.rand(N)-0.5).to(device)

# print simulation setting
print('\tN: %d' % N)
print('\tg: %.3f' % g)
print('\tp: %.3f' % p)
print('\tnRec2Out: %d'% nRec2Out)
print('\talpha: %.3f' % alpha)
print('\tnsecs: %d' % nsecs)
print('\tnsecs for train: %d' % nsecs_train)
print('\tlearn_every: %d' % learn_every)

simtime = np.arange(0,nsecs,dt)
simtime_len = len(simtime)
simtime_train_len = int(simtime_len/nsecs*nsecs_train)

amps = 1.3 / np.array([1.0, 2.0, 6.0, 3.0])
freqs = 1/60 * np.array([1.0, 2.0, 3.0, 4.0])
ft = np.zeros_like(simtime)
for amp, freq in zip(amps, freqs):
	ft += amp*np.sin(np.pi*freq*simtime)
ft = ft/1.5
ft = torch.Tensor(ft).to(device)

wo_len = torch.zeros(simtime_len).to(device)
zt = torch.zeros(simtime_len).to(device)
x0 = 0.5*torch.randn(N).to(device)
z0 = 0.5*torch.randn(1).to(device)

x = x0 
r = torch.tanh(x)
z = z0

frame_id = 0
P = (1.0/alpha)*torch.eye(nRec2Out).to(device)
torch.cuda.synchronize()
print(f'matrix init takes {time.time()-t0:.3f} s')
torch.cuda.synchronize()
t0 = time.time()
for ti in np.arange(len(simtime)):
    # sim, so x(t) and r(t) are created.
	# delayed and nonlinear distortion
	# if ti >=10:
	# 	x += dt*(-x+M_spa@r+wf*1.3*np.tanh(np.sin(np.pi*zt[ti-10])))
	# else:
	# 	x += dt * (-x + M_spa @ r)
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
	wo_len[ti] = torch.sqrt(wo@wo)	

torch.cuda.synchronize()
print(f'evolve dynamics takes {time.time()-t0:.3f} s')
# save outputs
# ------------
ft_cpu = np.array(ft.cpu(), dtype=float)
wo_len_cpu = np.array(wo_len.cpu(), dtype=float)
zt_cpu = np.array(zt.cpu(), dtype=float)
np.savez('test_tmp_data.npz', ft=ft_cpu, wo_len=wo_len_cpu, zt = zt_cpu)

# print training error
error_avg = torch.sum(torch.abs(zt[:simtime_train_len]-ft[:simtime_train_len]))/simtime_train_len
print(f'Training MAE:  {error_avg.cpu():3f}')    

error_avg = torch.sum(torch.abs(zt[simtime_train_len:]-ft[simtime_train_len:]))/simtime_train_len
print(f'Testing MAE:  {error_avg.cpu():3f}')

# save final frame as figure
fig2, ax2 = plt.subplots(2,1,figsize=(12,10))
ax2[0].plot(simtime, ft_cpu, color='green', label='f')
ax2[0].plot(simtime, zt_cpu, color='red', label='z')
ax2[0].set_title('Training')
ax2[0].set_xlabel('Time')
ax2[0].set_ylabel(r'$f$ and $z$')
ax2[0].axvline(simtime[simtime_train_len],color='cyan')
ax2[0].legend()

ax2[1].plot(simtime, wo_len_cpu, label='|w|')[0]
ax2[1].set_xlabel('Time')
ax2[1].set_ylabel(r'|$w$|')
ax2[1].axvline(simtime[simtime_train_len],color='cyan')
ax2[1].legend()
plt.tight_layout()

plt.savefig('Figure3.png')