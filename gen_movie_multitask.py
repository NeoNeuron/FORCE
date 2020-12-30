# gen_movie.py
#
# Generate movie of training and testing dynamics for multi-task training.
#
# Author: Kai Chen

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['font.size'] = 14
mpl.rcParams['font.weight'] = 'bold'

t0 = time.time()

# load data 
# ------------
raw_data = np.load('trained_net_10_test_data.npz')
fts = raw_data['fts']
zts = raw_data['zts']

dt = 0.1
simtime_len = fts.shape[1]
simtime = np.arange(simtime_len) * dt

# generate leaning animation:
# ----
t0 = time.time()
# prepare image container
video_duration = 5000	# ms
frame_interval = 100
nframe = int(len(simtime)/frame_interval)
frame_interval_time = int(video_duration/nframe)
training_imgs = [[] for _ in range(fts.shape[0])]

for k in range(fts.shape[0]):
    for i in range(nframe):
        training_imgs[k].append(zts[k][:i*frame_interval].copy())
    training_imgs[k].append(zts[k].copy())

xmax = simtime_len*dt

fig, ax = plt.subplots(fts.shape[0],1, figsize=(10,20), sharex=True)
lines = []
for idx, ax_i in enumerate(ax):
	ax_i.plot(simtime, fts[idx], color='green', label='f')
	line = ax_i.plot(simtime, zts[idx], color='red', label='z')[0]
	lines.append(line)
	ax_i.set_xlim(0, xmax)
	ax_i.set_title(f'Test {idx:d}')
	ax_i.legend(loc=2)	
	ax_i.set_ylabel(r'$f$ and $z$')
ax[-1].set_xlabel('Time')

plt.tight_layout()

def init():  # only required for blitting to give a clean slate.
	for idx, line in enumerate(lines):
		line.set_ydata([np.nan] * len(zts[idx]))

def animate(i):
	for idx, line in enumerate(lines):
		line.set_data(np.arange(len(training_imgs[idx][i]))*dt, training_imgs[idx][i])

ani = FuncAnimation(fig, animate, interval=frame_interval_time, frames=nframe)

ani.save('training_dynamic.mp4')

print(f'generating animation takes {time.time()-t0:.3f} s')