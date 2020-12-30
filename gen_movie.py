# gen_movie.py
#
# Generate movie of training and testing dynamics.
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
raw_data = np.load('test_tmp_data.npz')
ft = raw_data['ft']
wo_len = raw_data['wo']
zt = raw_data['zt']

dt = 0.1
simtime_len = len(ft)
simtime = np.arange(simtime_len) * dt
# simtime_train_len = int(simtime_len/nsecs*nsecs_train)

# generate leaning animation:
# ----
t0 = time.time()
# prepare image container
video_duration = 5000	# ms
frame_interval = 100
nframe = int(len(simtime)/frame_interval)
frame_interval_time = int(video_duration/nframe)
training_imgs = np.zeros((nframe,simtime_len))
weight_imgs = np.zeros((nframe,simtime_len))
for i in range(nframe):
	training_imgs[i,:i*frame_interval] = zt[:i*frame_interval]
	weight_imgs[i,:i*frame_interval] = wo_len[:i*frame_interval]

training_imgs = np.vstack((training_imgs, zt.reshape((1,-1))))
weight_imgs = np.vstack((weight_imgs, zt.reshape((1,-1))))

fig, ax = plt.subplots(2,1, figsize=(14,10))
ax[0].plot(simtime, ft, color='green', label='f')
line1 = ax[0].plot(simtime, training_imgs[0], color='red', label='z')[0]
# ax[0].axvline(simtime[simtime_train_len],color='cyan', label='End of Training')
ax[0].set_title('Training')
ax[0].legend(loc=2)	
ax[0].set_xlabel('Time')
ax[0].set_ylabel(r'$f$ and $z$')

line2 = ax[1].plot(simtime, wo_len, label='|w|')[0]
# ax[1].axvline(simtime[simtime_train_len],color='cyan', label='End of Training')
ax[1].set_xlabel('Time')
ax[1].set_ylabel(r'|$w$|')
ax[1].legend(loc=2)
plt.tight_layout()

# def init():  # only required for blitting to give a clean slate.
#     line2.set_ydata([np.nan] * len(wo_len))

def animate(i):
	line1.set_ydata(training_imgs[i])
	line2.set_ydata(weight_imgs[i])

ani = FuncAnimation(fig, animate, interval=frame_interval_time, frames=nframe)

ani.save('training_dynamic.mp4')

print(f'generating animation takes {time.time()-t0:.3f} s')