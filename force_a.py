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

mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['font.size'] = 14
mpl.rcParams['font.weight'] = 'bold'

t0 = time.time()
N = 1000
p = 0.1
g = 1.5		# g greater than 1 leads to chaotic networks.
alpha = 1.0
nsecs = 2000
nsecs_train = 1000
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

wo_len = np.zeros(simtime_len)    
zt = np.zeros(simtime_len)
x0 = 0.5*np.random.randn(N)
z0 = 0.5*np.random.randn()

x = x0.copy()
r = np.tanh(x)
z = z0.copy()

# prepare image container
video_duration = 5000	# ms
frame_interval = 100
nframe = int(len(simtime)/frame_interval)
frame_interval_time = int(video_duration/nframe)
training_imgs = np.zeros((nframe,simtime_len))
weight_imgs = np.zeros((nframe,simtime_len))

frame_id = 0
P = (1.0/alpha)*np.eye(nRec2Out)
print(f'matrix init takes {time.time()-t0:.3f} s')
t0 = time.time()
for ti in np.arange(len(simtime)):
	if ti % frame_interval == 0:
		# record current training state
		training_imgs[frame_id] = zt
		weight_imgs[frame_id] = wo_len
		frame_id += 1
    # sim, so x(t) and r(t) are created.
	# delayed and nonlinear distortion
	# if ti >=10:
	# 	x += dt*(-x+M_spa@r+wf*1.3*np.tanh(np.sin(np.pi*zt[ti-10])))
	# else:
	# 	x += dt * (-x + M_spa @ r)
	x += dt * (-x + M_spa @ r + wf*z)
	r = np.tanh(x)
	z = wo @ r
    
	if ((ti+1) % learn_every) == 0 and ti < simtime_train_len:
		# update inverse correlation matrix
		k = P @ r
		rPr = r @ k
		c = 1.0/(1.0 + rPr)
		P -= c * np.outer(k,k)
		
		# update the error for the linear readout
		e = z-ft[ti]
		
		# update the output weights
		dw = -e*k*c	
		wo += dw
    
    # Store the output of the system.
	zt[ti] = z
	wo_len[ti] = np.sqrt(wo@wo)	

print(f'evolve dynamics takes {time.time()-t0:.3f} s')
# adding last frame
training_imgs = np.append(training_imgs,[zt], axis=0)
weight_imgs = np.append(weight_imgs,[wo_len], axis=0)

t0 = time.time()
fig, ax = plt.subplots(2,1, figsize=(14,10))
ax[0].plot(simtime, ft, color='green', label='f')
line1 = ax[0].plot(simtime, training_imgs[0], color='red', label='z')[0]
ax[0].axvline(simtime[simtime_train_len],color='cyan', label='End of Training')
ax[0].set_title('Training')
ax[0].legend(loc=2)	
ax[0].set_xlabel('Time')
ax[0].set_ylabel(r'$f$ and $z$')

line2 = ax[1].plot(simtime, wo_len, label='|w|')[0]
ax[1].axvline(simtime[simtime_train_len],color='cyan', label='End of Training')
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

# print training error
error_avg = np.sum(np.abs(zt[:simtime_train_len]-ft[:simtime_train_len]))/simtime_train_len
print(f'Training MAE:  {error_avg:3f}')    

error_avg = np.sum(np.abs(zt[simtime_train_len:]-ft[simtime_train_len:]))/simtime_train_len
print(f'Testing MAE:  {error_avg:3f}')

# save final frame as figure
fig2, ax2 = plt.subplots(2,1,figsize=(12,10))
ax2[0].plot(simtime, ft, color='green', label='f')
ax2[0].plot(simtime, zt, color='red', label='z')
ax2[0].set_title('Training')
ax2[0].set_xlabel('Time')
ax2[0].set_ylabel(r'$f$ and $z$')
ax2[0].axvline(simtime[simtime_train_len],color='cyan')
ax2[0].legend()

ax2[1].plot(simtime, wo_len, label='|w|')[0]
ax2[1].set_xlabel('Time')
ax2[1].set_ylabel(r'|$w$|')
ax2[1].axvline(simtime[simtime_train_len],color='cyan')
ax2[1].legend()
plt.tight_layout()

plt.savefig('FORCE_Type_A.png')