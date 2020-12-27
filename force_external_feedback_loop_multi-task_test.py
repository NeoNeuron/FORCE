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
from matplotlib.animation import FuncAnimation

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

fts = [gen_target(amp, freqs, simtime) for amp in amps]
single_len = len(fts[0])

input_bias = np.repeat(input_bias_set.reshape((input_bias_set.shape[0], 1, input_bias_set.shape[1])), 
						simtime_len, axis=1)

x0 = 0.5*np.random.randn(N)
z0 = 0.5*np.random.randn()

# prepare image container
video_duration = 10000	# ms
frame_interval = 200
nframe = int(len(simtime)/frame_interval)
frame_interval_time = int(video_duration/nframe)
training_imgs = [[] for _ in range(len(input_bias_set))]

zt_total = []

for iter in range(len(input_bias_set)):
	zt = []
	x = x0 
	r = np.tanh(x)
	z = z0
	for ti in np.arange(simtime_len):
		if ti % frame_interval == 0:
			# record current training state
			training_imgs[iter].append(zt.copy())
		# sim, so x(t) and r(t) are created.
		x += dt * (-x + M @ r + wf*z + J_GI@input_bias[iter,ti])
		r = np.tanh(x)
		z = wo @ r
	
		# Store the output of the system.
		zt.append(z)
	# adding last frame
	training_imgs[iter].append(zt.copy())
	zt_total.append(np.array(zt))

# generate movie
xmax = nsecs

fig, ax = plt.subplots(len(input_bias_set),1, figsize=(10,20), sharex=True)
lines = []
for idx, ax_i in enumerate(ax):
	ax_i.plot(simtime, fts[idx], color='green', label='f')
	line = ax_i.plot(simtime, zt_total[idx], color='red', label='z')[0]
	lines.append(line)
	ax_i.set_xlim(0, xmax)
	ax_i.set_title(f'Test {idx:d}')
	ax_i.legend(loc=2)	
	ax_i.set_ylabel(r'$f$ and $z$')
ax[-1].set_xlabel('Time')

# plt.tight_layout()

def init():  # only required for blitting to give a clean slate.
	for line in lines:
		line.set_ydata([np.nan] * len(zt_total[0]))

def animate(i):
	for idx, line in enumerate(lines):
		line.set_data(np.arange(len(training_imgs[idx][i]))*dt, training_imgs[idx][i])

ani = FuncAnimation(fig, animate, init_func = init, interval=frame_interval_time, frames=nframe)

ani.save('test_dynamic.mp4')

# save final frame as figure
fig2, ax2 = plt.subplots(len(input_bias_set),1,figsize=(10,20),sharex=True)
for idx, ax_i in enumerate(ax2):
	ax_i.plot(simtime, fts[idx], color='green', label='f')
	ax_i.plot(simtime, zt_total[idx], color='red', label='z')
	ax_i.set_title(f'Test {idx:d}')
	ax_i.set_ylabel(r'$f$ and $z$')
	ax_i.legend()
ax2[-1].set_xlabel('Time')

plt.tight_layout()

plt.savefig('Figure3_test.png')