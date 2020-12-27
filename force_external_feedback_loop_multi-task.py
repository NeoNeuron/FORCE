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
import itertools
import time

mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['font.size'] = 14
mpl.rcParams['font.weight'] = 'bold'

N = 1000
n_input = 100
p = 0.1
g = 1.5		# g greater than 1 leads to chaotic networks.
alpha = 0.0125
nsecs_train = 1920
dt = 0.1
learn_every = 2

# sparse matrix M
M = np.random.rand(N,N)
M = (M<p).astype(int)
scale = 1.0/np.sqrt(p*N)
M = M * g*scale * np.random.randn(N,N)
M_spa = csr_matrix(M)

nRec2Out = N
wo = np.zeros(nRec2Out)
dw = np.zeros(nRec2Out)
wf = 2.0*(np.random.rand(N)-0.5)

# generate connectivity matrix for control inputs
J_GI = np.zeros((N, n_input))
col_indices = np.random.randint(n_input, size=N)
J_GI[np.arange(N, dtype=int), col_indices] = np.random.randn(N)
J_GI_spa = csr_matrix(J_GI)

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
n_options = 2
options = options[:n_options]
training_len = simtime_len * len(options)
amps = [1.3/np.array(item) for item in options]
# amps1 = 1.3 / np.array([1.0, 2.0, 6.0, 3.0])
# amps2 = 1.3 / np.array([6.0, 1.0, 2.0, 3.0])
freqs = 1/60 * np.array([1.0, 2.0, 3.0, 4.0])

fts_train = [gen_target(amp, freqs, simtime) for amp in amps]
# ft2 = gen_target(amps2, freqs, simtime)
fts_test = [gen_target(amp, freqs, np.arange(0,720,dt)) for amp in amps]
# ft2_single = gen_target(amps2, freqs, np.arange(0,360,dt))
single_len = len(fts_test[0])
ft = np.hstack(fts_train)

input_bias_set = 1.6*(np.random.rand(len(options), n_input)-0.5)
input_bias = np.repeat(input_bias_set, simtime_len, axis=0)

# generate test samples
# random_sample = np.random.randint(len(options), size=20)
random_sample = np.repeat(np.arange(n_options),2)
for val in random_sample:
	ft = np.hstack((ft, fts_test[val])) 
	input_bias = np.vstack((input_bias, np.tile(input_bias_set[val], (single_len, 1))))

simtime_len = len(ft)
simtime = np.arange(simtime_len) * dt

# wo_len = np.zeros(simtime_len)
# zt = np.zeros(simtime_len)
wo_len = []
zt = []
x0 = 0.5*np.random.randn(N)
z0 = 0.5*np.random.randn()

x = x0 
r = np.tanh(x)
z = z0

# prepare image container
video_duration = 100000	# ms
frame_interval = 200
nframe = int(len(simtime)/frame_interval)
frame_interval_time = int(video_duration/nframe)
# training_imgs = np.zeros((nframe,simtime_len))
# weight_imgs = np.zeros((nframe,simtime_len))
training_imgs = []
weight_imgs = []

t0 = time.time()
P = (1.0/alpha)*np.eye(nRec2Out)
for ti in np.arange(simtime_len):
	if ti % frame_interval == 0:
		# record current training state
		training_imgs.append(zt.copy())
		weight_imgs.append(wo_len.copy())
		# frame_id += 1
    # sim, so x(t) and r(t) are created.
	x += dt * (-x + M_spa @ r + wf*z + J_GI_spa@input_bias[ti])
	r = np.tanh(x)
	z = wo @ r
    
	if ((ti+1) % learn_every) == 0 and ti < training_len:
		# update inverse correlation matrix
		k = P @ r
		rPr = r @ k
		c = 1.0/(1.0 + rPr)
		P -= c *np.outer(k,k)
		
		# update the error for the linear readout
		e = z-ft[ti]
		
		# update the output weights
		dw = -e*k*c	
		wo += dw
    
    # Store the output of the system.
	# zt[ti] = z
	# wo_len[ti] = np.sqrt(wo@wo)	
	zt.append(z)
	wo_len.append(np.sqrt(wo@wo))

print(f'it tooks {time.time()-t0:5.3f} s')
# adding last frame
training_imgs.append(zt.copy())
weight_imgs.append(wo_len.copy())
zt = np.array(zt)
wo_len = np.array(wo_len)

x_range = 2000
xmax = x_range
dx = x_range / 2

fig, ax = plt.subplots(2,1, figsize=(20,10))
ax[0].plot(simtime, ft, color='green', label='f')
line1 = ax[0].plot(simtime, zt, color='red', label='z')[0]
ax[0].axvline(simtime[training_len-1],color='cyan', label='End of Training')
ax[0].set_xlim(0, xmax)
ax[0].set_title('Training')
ax[0].legend(loc=2)
ax[0].set_xlabel('Time')
ax[0].set_ylabel(r'$f$ and $z$')

line2 = ax[1].plot(simtime, wo_len, label='|w|')[0]
ax[1].axvline(simtime[training_len-1],color='cyan', label='End of Training')
ax[1].set_xlim(0, xmax)
ax[1].set_xlabel('Time')
ax[1].set_ylabel(r'|$w$|')
ax[1].legend(loc=2)
plt.tight_layout()

def init():  # only required for blitting to give a clean slate.
    line1.set_ydata([np.nan] * len(zt))
    line2.set_ydata([np.nan] * len(wo_len))

def animate(i):
	line1.set_data(np.arange(len(training_imgs[i]))*dt, training_imgs[i])
	line2.set_data(np.arange(len(weight_imgs[i]))*dt, weight_imgs[i])
	global xmax
	if len(training_imgs[i]) > xmax / dt:
		xmax += dx
		ax[0].set_xlim(xmax - x_range, xmax)
		ax[1].set_xlim(xmax - x_range, xmax)


ani = FuncAnimation(fig, animate, init_func = init, interval=frame_interval_time, frames=nframe)

ani.save('training_dynamic.mp4')

# print training error
error_avg = np.sum(np.abs(zt[:int(simtime_len/2)]-ft[:int(simtime_len/2)]))/simtime_len*2
print(f'Training MAE:  {error_avg:3f}')    

error_avg = np.sum(np.abs(zt[int(simtime_len/2):]-ft[int(simtime_len/2):]))/simtime_len*2
print(f'Testing MAE:  {error_avg:3f}')

# save final frame as figure
fig2, ax2 = plt.subplots(2,1,figsize=(20,10))
ax2[0].plot(simtime, ft, color='green', label='f')
ax2[0].plot(simtime, zt, color='red', label='z')
ax2[0].set_title('Training')
ax2[0].set_xlabel('Time')
ax2[0].set_ylabel(r'$f$ and $z$')
ax2[0].axvline(simtime[training_len],color='cyan')
ax2[0].legend()

ax2[1].plot(simtime, wo_len, label='|w|')[0]
ax2[1].set_xlabel('Time')
ax2[1].set_ylabel(r'|$w$|')
ax2[1].axvline(simtime[training_len],color='cyan')
ax2[1].legend()
plt.tight_layout()

plt.savefig('Figure3.png')
np.savez(f'trained_net_{n_options:d}.npz', Jgg=M, Jgi=J_GI, I = input_bias_set, w=wo, wf = wf, options = options)