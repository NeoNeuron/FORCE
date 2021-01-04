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
# 	pure numpy version: 	96.001 s
#	PyTorch CUDA version: 	16.001 s
#	PyTorch CPU version: 	31.121 s


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools
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
n_input = 100
p = 0.1
g = 1.5		# g greater than 1 leads to chaotic networks.
alpha = 0.0125
nsecs_train = 2400
dt = 0.1
learn_every = 2
test_toggle = False

# sparse matrix M
M = np.random.rand(N,N)
M = (M<p).astype(int)
scale = 1.0/np.sqrt(p*N)
M = M * g*scale * np.random.randn(N,N)
M = torch.Tensor(M).to(device)

nRec2Out = N
wo = torch.zeros(nRec2Out).to(device)
dw = torch.zeros(nRec2Out).to(device)
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
print('\tnRec2Out: %d'% nRec2Out)
print('\talpha: %.3f' % alpha)
print('\tnsecs train: %d' % nsecs_train)
print('\tlearn_every: %d' % learn_every)

simtime = np.arange(0,nsecs_train,dt)
simtime_len = len(simtime)

def gen_target(amps, freqs, time):
	ft = np.zeros_like(time)
	for amp, freq in zip(amps, freqs):
		ft += amp*np.sin(np.pi*freq*time)
	ft = ft/1.5
	return ft

options = list(itertools.permutations([1.0,2.0,3.0,6.0],4))
n_options = 10
rg = np.random.Generator(np.random.MT19937(0))
choice = rg.choice(np.arange(len(options)), n_options, replace=False)
# options = options[:n_options]
options = [options[idx] for idx in choice]

# shifting order again
order = np.arange(n_options, dtype=int)
rg.shuffle(order)
options = [options[idx] for idx in order]

training_len = simtime_len * len(options)
amps = [1.3/np.array(item) for item in options]
freqs = 1/60 * np.array([1.0, 2.0, 3.0, 4.0])

fts_train = [gen_target(amp, freqs, simtime) for amp in amps]
fts_test = [gen_target(amp, freqs, np.arange(0,720,dt)) for amp in amps]
single_len = len(fts_test[0])
ft = np.hstack(fts_train)

input_bias_set = 1.6*(np.random.rand(len(options), n_input)-0.5)
input_bias = np.repeat(input_bias_set, simtime_len, axis=0)

# generate test samples
# ---------------------
if test_toggle:
	# random_sample = np.random.randint(len(options), size=20)
	random_sample = np.repeat(np.arange(n_options),2)
	for val in random_sample:
		ft = np.hstack((ft, fts_test[val])) 
		input_bias = np.vstack((input_bias, np.tile(input_bias_set[val], (single_len, 1))))

# convert to GPU memory
ft = torch.Tensor(ft).to(device)
input_bias = torch.Tensor(input_bias).to(device)

simtime_len = len(ft)
simtime = np.arange(simtime_len) * dt

wo_len = torch.zeros(simtime_len).to(device)
zt = torch.zeros(simtime_len).to(device)
x0 = 0.5*torch.randn(N).to(device)
z0 = 0.5*torch.randn(1).to(device)

x = x0 
r = torch.tanh(x)
z = z0

P = (1.0/alpha)*torch.eye(nRec2Out).to(device)
torch.cuda.synchronize()
print(f'matrix init takes {time.time()-t0:.3f} s')
torch.cuda.synchronize()
t0 = time.time()
for ti in np.arange(simtime_len):
    # sim, so x(t) and r(t) are created.
	x += dt * (-x + M @ r + wf*z + J_GI@input_bias[ti])
	r = torch.tanh(x)
	z = wo @ r
    
	if ((ti+1) % learn_every) == 0 and ti < training_len:
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
	wo_len[ti] = torch.sqrt(wo@wo)	

torch.cuda.synchronize()
print(f'evolve dynamics takes {time.time()-t0:.3f} s')

# save trained model parameters
ft_cpu = np.array(ft.cpu(), dtype=float)
wo_len_cpu = np.array(wo_len.cpu(), dtype=float)
zt_cpu = np.array(zt.cpu(), dtype=float)
np.savez('test_tmp_data_multitask.npz', ft=ft_cpu, wo_len=wo_len_cpu, zt = zt_cpu)
np.savez(f'trained_net_{n_options:d}.npz', Jgg=M.cpu(), Jgi=J_GI.cpu(), I = input_bias_set, w=wo.cpu(), wf = wf.cpu(), options = options, order = order)

# print training error
error_avg = torch.sum(torch.abs(zt[:int(simtime_len/2)]-ft[:int(simtime_len/2)]))/simtime_len*2
print(f'Training MAE:  {error_avg.cpu():3f}')    

error_avg = torch.sum(torch.abs(zt[int(simtime_len/2):]-ft[int(simtime_len/2):]))/simtime_len*2
print(f'Testing MAE:  {error_avg.cpu():3f}')

# save final frame as figure
fig2, ax2 = plt.subplots(2,1,figsize=(20,10))
ax2[0].plot(simtime, ft_cpu, color='green', label='f')
ax2[0].plot(simtime, zt_cpu, color='red', label='z')
ax2[0].set_title('Training')
ax2[0].set_xlabel('Time')
ax2[0].set_ylabel(r'$f$ and $z$')
ax2[0].axvline(simtime[training_len-1],color='cyan')
ax2[0].legend()

ax2[1].plot(simtime, wo_len_cpu, label='|w|')[0]
ax2[1].set_xlabel('Time')
ax2[1].set_ylabel(r'|$w$|')
ax2[1].axvline(simtime[training_len-1],color='cyan')
ax2[1].legend()
plt.tight_layout()

plt.savefig('Figure3.png')