# FORCE.py
#
# This function generates the sum of 4 sine waves in figure 2D using the architecture of figure 1A with the RLS
# learning rule.
#
# MATLAB vertion written by David Sussillo
# Modified by Kai Chen

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from multiprocessing import Pool
import time
from scipy.sparse import csr_matrix

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

# create sparse version of variables
M_spa = csr_matrix(M)
J_GI_spa = csr_matrix(J_GI)

N = M.shape[0]
n_input = J_GI.shape[1]
dt = 0.1
nRec2Out = N
nsecs = 1200
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

fts = [gen_target(amp, freqs, simtime) for amp in amps]
single_len = len(fts[0])

input_bias = np.repeat(input_bias_set.reshape((input_bias_set.shape[0], 1, input_bias_set.shape[1])), 
						simtime_len, axis=1)

def scan_corrcoef(x, y):
	buffer = np.zeros(int(120/dt))
	for i in np.arange(1, len(buffer)+1):
		buffer[i-1] = np.corrcoef(x[i:], y[:-i])[0,1]	
	return np.max(buffer)

def single_init_test(seed:int)->np.ndarray:
	# initialize randomness
	np.random.seed(seed)
	z0 = 0.5*np.random.randn()
	x0 = 0.5*np.random.randn(N)

	zt_total = []

	for iter in range(len(input_bias_set)):
		zt = []
		x = x0 
		r = np.tanh(x)
		z = z0
		for ti in np.arange(simtime_len):
			# sim, so x(t) and r(t) are created.
			x += dt * (-x + M_spa @ r + wf*z + J_GI_spa@input_bias[iter,ti])
			r = np.tanh(x)
			z = wo @ r
		
			# Store the output of the system.
			zt.append(z)
		# save last frame
		zt_total.append(np.array(zt))

	corr = np.zeros(len(fts))
	for idx, ft in enumerate(fts):
		corr[idx] = scan_corrcoef(ft[int(simtime_len/2):], zt_total[idx][int(simtime_len/2):])
	return corr


# Test single trial
# t0 = time.time()
# corr = single_init_test(100)
# print(f'it tooks {time.time()-t0:5.3f} s')
# print(corr)

reps = 100
# multiprocessing 
pn = 10
p = Pool(pn)
result = [p.apply_async(func = single_init_test,args=(i,)) for i in range(reps)]
p.close()
p.join()
corr = np.zeros((reps, len(input_bias_set)))
i = 0
for res in result:
	corr[i,:] = res.get()
	i += 1
np.save('init_test_result.npy', corr)


fig, ax = plt.subplots(1,1)
ax.plot(np.arange(1, len(input_bias_set)+1), corr.mean(0), color='b', label='correlation', yerr=corr.std(0))
ax.set_ylabel('Correlation coefficient')
ax.legend(loc=2)
ax.set_xlabel('Index of input patterns')

plt.tight_layout()

plt.savefig('Figure3_test_init.png')