# FORCE.py
#
# This function generates the sum of 4 sine waves in figure 2D using the architecture of figure 1A with the RLS
# learning rule.
#
# MATLAB vertion written by David Sussillo
# Modified by Kai Chen
# Benchmarks:	
# 	10 pattern, 100 trials case
# 	pure numpy version:		582.897 s (40 processes)
# 	pure numpy version:		962.596 s (20 processes)
#	PyTorch CUDA version: 	663.807 s (20 processes)
#	PyTorch CPU version: 	  s
# Single task:
#	pure numpy version:     36.110 s
#	PyTorch CUDA version: 	19.850 s
#	PyTorch CPU version: 	28.660 s

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# from multiprocessing import Pool
import torch.multiprocessing as mp
import time
import torch

if torch.cuda.is_available(): 	   # device agnostic code
    device = torch.device('cuda')  # and modularity
else:                              #
    device = torch.device('cpu')   #
# device = torch.device('cpu')   #

mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['font.size'] = 14
mpl.rcParams['font.weight'] = 'bold'

# load parameters
params = np.load('trained_net_10.npz')
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

def scan_corrcoef(x:torch.Tensor, y:torch.Tensor)->float:
	# define corrcoef for cuda
	def corrcoef(x:torch.Tensor, y:torch.Tensor)->torch.Tensor:
		return ((x*y).mean() - x.mean()*y.mean())/x.std()/y.std()
	# scan across time delay
	buffer = np.zeros(int(120/dt))
	for i in np.arange(1, len(buffer)+1):
		buffer[i-1] = corrcoef(x[i:], y[:-i]).cpu()	
	return buffer[~np.isnan(buffer)].max()

def single_init_test(seed:int)->np.ndarray:
	# initialize randomness
	torch.manual_seed(seed)
	z0 = 0.5*torch.randn(1).to(device)
	x0 = 0.5*torch.randn(N).to(device)

	zt_total = []

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
		# save last frame
		zt_total.append(zt)

	corr = np.zeros(len(fts))
	for idx, ft in enumerate(fts):
		corr[idx] = scan_corrcoef(ft[int(simtime_len/2):], zt_total[idx][int(simtime_len/2):])
	return corr

if __name__  == '__main__':
	# Test single trial
	torch.cuda.synchronize()
	t0 = time.time()
	mp.set_start_method('spawn')
	reps = 100
	# multiprocessing 
	pn = 20
	p = mp.Pool(pn)
	result = [p.apply_async(func = single_init_test,args=(i,)) for i in range(reps)]
	p.close()
	p.join()
	corr = np.zeros((reps, len(input_bias_set)))
	i = 0
	for res in result:
		corr[i,:] = res.get()
		i += 1
	np.save('init_test_result.npy', corr)

	torch.cuda.synchronize()
	print(f'evolve dynamics takes {time.time()-t0:.3f} s')

	fig, ax = plt.subplots(1,1)
	ax.boxplot(corr)
	ax.set_ylabel('Correlation coefficient')
	ax.set_xlabel('Index of input patterns')

	plt.tight_layout()

	plt.savefig('Figure3_test_init.png')