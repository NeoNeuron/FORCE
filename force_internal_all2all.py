# FORCE_INTERNAL_ALL2ALL.py
#
# This function generates the sum of 4 sine waves in figure 2D using the arcitecture of figure 1C (all-to-all
# connectivity) with the RLS learning rule.  The all-2-all connectivity allows a large optimization, in that we can
# maintain a single inverse correlation matrix for the network.  It's also not as a hard a learning problem as internal
# learning with sparse connectivity because there are no approximations of the eigenvectors of the correlation matrices,
# as there would be if this was sparse internal learning.  Note that there is no longer a feedback loop from the output
# unit.
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

N = 1000
p = 1.0
g = 1.5		# g greater than 1 leads to chaotic networks.
alpha = 1.0e-0
nsecs = 600
dt = 0.1
learn_every = 2

scale = 1.0/np.sqrt(p*N)
M = np.random.randn(N,N) *g*scale

nRec2Out = N
wo = np.zeros(nRec2Out)
dw = np.zeros(nRec2Out)

print('   N: %d' % N)
print('   g: %.3f' % g)
print('   p: %.3f' % p)
print('   nRec2Out: %d'% nRec2Out)
print('   alpha: %.3f' % alpha)
print('   nsecs: %d' % nsecs)
print('   learn_every: %d' % learn_every)


simtime = np.arange(0,nsecs,dt)
simtime_len = len(simtime)
simtime2 = simtime + nsecs

amps = 0.7 / np.array([1.0, 2.0, 6.0, 3.0])
freqs = 1/60 * np.array([1.0, 2.0, 3.0, 4.0])
ft = np.zeros_like(simtime)
for amp, freq in zip(amps, freqs):
	ft += amp*np.sin(np.pi*freq*simtime)
ft = ft/1.5


wo_len = np.zeros(simtime_len)    
zt = np.zeros(simtime_len)
x0 = 0.5*np.random.randn(N)
z0 = 0.5*np.random.randn()

x = x0 
r = np.tanh(x)
z = z0 

# prepare image container
frame_interval = 100
nframe = int(len(simtime)/frame_interval)
training_imgs = np.zeros((nframe,simtime_len))
weight_imgs = np.zeros((nframe,simtime_len))

frame_id = 0
P = (1.0/alpha)*np.eye(nRec2Out)
for ti in np.arange(len(simtime)):
	if ti % frame_interval == 0:
		# record current training state
		training_imgs[frame_id] = zt
		weight_imgs[frame_id] = wo_len
		frame_id += 1
    # sim, so x(t) and r(t) are created.
	x += dt * (-x + M @ r)
	r = np.tanh(x)
	z = wo @ r
    
	if ((ti+1) % learn_every) == 0 and ti < len(simtime)/2:
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
		
		# update the internal weight matrix using the output's error
		M += np.tile(dw, (N, 1))
    
    # Store the output of the system.
	zt[ti] = z
	wo_len[ti] = np.sqrt(wo@wo)	

# adding last frame
training_imgs = np.append(training_imgs,[zt], axis=0)
weight_imgs = np.append(weight_imgs,[wo_len], axis=0)

fig, ax = plt.subplots(2,1, figsize=(14,10))
ax[0].plot(simtime, ft, color='green', label='f')
line1 = ax[0].plot(simtime, training_imgs[0], color='red', label='z')[0]
ax[0].axvline(simtime[int(simtime_len/2)],color='cyan', label='End of Training')
ax[0].set_title('Training')
ax[0].legend(loc=2)	
ax[0].set_xlabel('Time')
ax[0].set_ylabel(r'$f$ and $z$')

line2 = ax[1].plot(simtime, wo_len, label='|w|')[0]
ax[1].axvline(simtime[int(simtime_len/2)],color='cyan', label='End of Training')
ax[1].set_xlabel('Time')
ax[1].set_ylabel(r'|$w$|')
ax[1].legend(loc=2)
plt.tight_layout()

# def init():  # only required for blitting to give a clean slate.
#     line2.set_ydata([np.nan] * len(wo_len))

def animate(i):
	line1.set_ydata(training_imgs[i])
	line2.set_ydata(weight_imgs[i])

ani = FuncAnimation(fig, animate, interval=60, frames=nframe)

ani.save('training_dynamic.mp4')

error_avg = np.sum(np.abs(zt[:int(simtime_len/2)]-ft[:int(simtime_len/2)]))/simtime_len*2
print(f'Training MAE:  {error_avg:3f}')    

error_avg = np.sum(np.abs(zt[int(simtime_len/2):]-ft[int(simtime_len/2):]))/simtime_len*2
print(f'Testing MAE:  {error_avg:3f}')

fig2, ax2 = plt.subplots(2,1,figsize=(12,10))
ax2[0].plot(simtime, ft, color='green', label='f')
ax2[0].plot(simtime, zt, color='red', label='z')
ax2[0].set_title('Training')
ax2[0].set_xlabel('Time')
ax2[0].set_ylabel(r'$f$ and $z$')
ax2[0].axvline(simtime[int(simtime_len/2)],color='cyan')
ax2[0].legend()

ax2[1].plot(simtime, wo_len, label='|w|')[0]
ax2[1].set_xlabel('Time')
ax2[1].set_ylabel(r'|$w$|')
ax2[1].axvline(simtime[int(simtime_len/2)],color='cyan')
ax2[1].legend()
plt.tight_layout()

plt.savefig('Figure3.png')