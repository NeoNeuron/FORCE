# FORCE_INTERNAL_SPARSE.py
#
# This function generates the sum of 4 sine waves in figure 2D using the architecture of figure 1C (sparse connectivity)
# with the RLS learning rule.  We only implement one loop, otherwise we'd be here all week.  Literally.  This is because
# in the case of sparse connectivity, we don't have the option (and the optimization) to use the same inverse
# correlation matrix for all neurons, as we did n force_internal_all2all.m.  Rather, we'd have to have order(N) NxN
# inverse correlation matrices, and clearly that won't fly.  
#
# So implementing only one loop, the script demonstrates that a separate feedback loop that is not attached to the
# output unit can be trained using the output's error, even though the output and the control unit (the one that feed's
# its signal back into the network) do not share a single pre-synaptic input.  This principle is used heavily in both
# architectures in figure 1B and 1C for the real examples shown in the paper.
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

N = 2000
p = 0.1
g = 1.5		# g greater than 1 leads to chaotic networks.
alpha = 1.0e-0
nsecs = 2880
dt = 0.1
learn_every = 2

# sparse matrix M
M = np.random.rand(N,N)
M = (M<p).astype(int)
scale = 1.0/np.sqrt(p*N)
M = M * g*scale * np.random.randn(N,N)
M_spa = csr_matrix(M)

nRec2Out = int(N/2)
nRec2Control = int(N/2)

# Allow output and control units to start with different ICs.  If you set beta greater than zero, then y will look
# different than z but still drive the network with the appropriate frequency content (because it will be driven with
# z).  A value of beta = 0 shows that the learning rules produce extremely similar signals for both z(t) and y(t),
# despite having no common pre-synaptic inputs.  Keep in mind that the vector norm of the output weights is 0.1-0.2 when
# finished, so if you make beta too big, things will eventually go crazy and learning won't converge.
#beta = 0.1;	
beta = 0.0
wo = beta*np.random.randn(nRec2Out)/np.sqrt(nRec2Out)
dwo = np.zeros(nRec2Out)
wc = beta*np.zeros(nRec2Control)/np.sqrt(nRec2Control)
dwc = np.zeros(nRec2Control)

wf = 2.0*(np.random.rand(N)-0.5)

print('\tN: %d' % N)
print('\tg: %.3f' % g)
print('\tp: %.3f' % p)
print('\tnRec2Out: %d'% nRec2Out)
print('\tnRec2Control: %d'% nRec2Control)
print('\talpha: %.3f' % alpha)
print('\tnsecs: %d' % nsecs)
print('\tlearn_every: %d' % learn_every)


simtime = np.arange(0,nsecs,dt)
simtime_len = len(simtime)
simtime2 = simtime + nsecs

amps = 1.3 / np.array([1.0, 2.0, 6.0, 3.0])
freqs = 1/60 * np.array([1.0, 2.0, 3.0, 4.0])
ft = np.zeros_like(simtime)
for amp, freq in zip(amps, freqs):
	ft += amp*np.sin(np.pi*freq*simtime)
ft = ft/1.5


wo_len = np.zeros(simtime_len)    
wc_len = np.zeros(simtime_len)    
zt = np.zeros(simtime_len)
yt = np.zeros(simtime_len)
x0 = 0.5*np.random.randn(N)
z0 = 0.5*np.random.randn()
y0 = 0.5*np.random.randn()

x = x0 
r = np.tanh(x)
z = z0 
y = y0 

# prepare image container
frame_interval = 100
nframe = int(len(simtime)/frame_interval)
z_imgs = np.zeros((nframe,simtime_len))
y_imgs = np.zeros((nframe,simtime_len))
wo_imgs = np.zeros((nframe,simtime_len))
wc_imgs = np.zeros((nframe,simtime_len))

t0 = time.time()
frame_id = 0
Pz = (1.0/alpha)*np.eye(nRec2Out)
Py = (1.0/alpha)*np.eye(nRec2Control)
for ti in np.arange(len(simtime)):
	if ti % frame_interval == 0:
		# record current training state
		z_imgs[frame_id] = zt
		y_imgs[frame_id] = yt
		wo_imgs[frame_id] = wo_len
		wc_imgs[frame_id] = wc_len
		frame_id += 1
    # sim, so x(t) and r(t) are created.
	x += dt * (-x + M_spa @ r + wf*y)
	r = np.tanh(x)
	rz = r[:nRec2Out]
	ry = r[nRec2Out:]
	z = wo @ rz
	y = wc @ ry
    
	if ((ti+1) % learn_every) == 0 and ti < len(simtime)/2:
		# update inverse correlation matrix
		kz = Pz @ rz
		rPrz = rz @ kz
		cz = 1.0/(1.0 + rPrz)
		Pz -= cz * np.outer(kz,kz)
		
		# update the error for the linear readout
		e = z-ft[ti]
		
		# update the output weights
		dwo = -e*kz*cz	
		wo += dwo
		
		# update inverse correlation matrix for the control unit
		ky = Py @ ry
		rPry = ry @ ky
		cy = 1.0/(1.0 + rPry)
		Py -= cy * np.outer(ky,ky)
		
		# update the internal weight matrix using the output's error
		dwc = -e*ky*cy	
		wc += dwc
    
    # Store the output of the system.
	zt[ti] = z
	yt[ti] = y
	wo_len[ti] = np.sqrt(wo@wo)	
	wc_len[ti] = np.sqrt(wc@wc)	

print(f'it takes {time.time()-t0:.3f} s')

# adding last frame
z_imgs = np.append(z_imgs,[zt], axis=0)
y_imgs = np.append(y_imgs,[yt], axis=0)
wo_imgs = np.append(wo_imgs,[wo_len], axis=0)
wc_imgs = np.append(wc_imgs,[wc_len], axis=0)

fig, ax = plt.subplots(2,1, figsize=(14,10))
ax[0].plot(simtime, ft, color='green', label='f')
line11 = ax[0].plot(simtime, z_imgs[0], color='r', label='z')[0]
line12 = ax[0].plot(simtime, y_imgs[0], color='m', label='y')[0]
ax[0].axvline(simtime[int(simtime_len/2)],color='cyan', label='End of Training')
ax[0].set_title('Training')
ax[0].legend(loc=2)	
ax[0].set_xlabel('Time')
ax[0].set_ylabel(r'$f$ and $z$')

line21 = ax[1].plot(simtime, wo_len, label=r'$|w_o|$')[0]
line22 = ax[1].plot(simtime, wc_len, label=r'$|w_c|$')[0]
ax[1].axvline(simtime[int(simtime_len/2)],color='cyan', label='End of Training')
ax[1].set_xlabel('Time')
ax[1].set_ylabel(r'|$w$|')
ax[1].legend(loc=2)
plt.tight_layout()

# def init():  # only required for blitting to give a clean slate.
#     line2.set_ydata([np.nan] * len(wo_len))

def animate(i):
	line11.set_ydata(y_imgs[i])
	line12.set_ydata(z_imgs[i])
	line21.set_ydata(wo_imgs[i])
	line22.set_ydata(wc_imgs[i])

ani = FuncAnimation(fig, animate, interval=60, frames=nframe)

ani.save('training_dynamic_internal_sparse.mp4')

error_avg = np.sum(np.abs(zt[:int(simtime_len/2)]-ft[:int(simtime_len/2)]))/simtime_len*2
print(f'Training MAE:  {error_avg:3f}')    

error_avg = np.sum(np.abs(zt[int(simtime_len/2):]-ft[int(simtime_len/2):]))/simtime_len*2
print(f'Testing MAE:  {error_avg:3f}')

fig2, ax2 = plt.subplots(2,1,figsize=(12,10))
ax2[0].plot(simtime, ft, color='green', label='f')
ax2[0].plot(simtime, zt, color='red', label='z')
ax2[0].plot(simtime, yt, color='m', label='y')
ax2[0].set_title('Training')
ax2[0].set_xlabel('Time')
ax2[0].set_ylabel(r'$f$, $z$ and $y$')
ax2[0].axvline(simtime[int(simtime_len/2)],color='cyan')
ax2[0].legend()

ax2[1].plot(simtime, wo_len, label=r'$|w_o|$')[0]
ax2[1].plot(simtime, wc_len, label=r'$|w_c|$')[0]
ax2[1].set_xlabel('Time')
ax2[1].set_ylabel(r'|$w$|')
ax2[1].axvline(simtime[int(simtime_len/2)],color='cyan')
ax2[1].legend()
plt.tight_layout()

plt.savefig('Figure3_internal_sparse.png')