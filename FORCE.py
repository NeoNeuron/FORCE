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
#		pure numpy version: 	96.001 s
#		PyTorch CUDA version: 	16.001 s
#		PyTorch CPU version: 	31.121 s
#	10 pattern case
#		PyTorch CUDA version: 	59.001 s


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import torch
import json
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['font.size'] = 16

def RLS_inplace(w, error, P, r):
	# update inverse correlation matrix
	k = P @ r
	# rPr = r @ k
	c = 1.0/(1.0 + r @ k)
	P -= c * (k.reshape((-1,1)) @ k.reshape((1,-1)))
	# update the output weights
	# dw = -error*k*c	
	# w += dw
	w -= error*k*c
	return w, P

def LMS_inplace(w, error, eta, r):
	# update the output weights
	w -= eta*error*r
	return w

class RNN:
	def __init__(self, 
		N:int, 
		N_readout:int=1,
		W_rec:torch.Tensor=None,
		tau:float=20,
		W_readout:torch.Tensor=None,
		W_feedback:torch.Tensor=None,
		activation:callable=torch.tanh,
		*args, **kwargs,
	):
		self.N = N
		self.x = torch.zeros(N, dtype=torch.float32).to(device)
		self.r = self.x.clone().to(device)
		self.z = torch.zeros(N_readout, dtype=torch.float32).to(device)
		if W_rec is None:
			self.W_rec = torch.zeros((N,N), dtype=torch.float32).to(device)
		else:
			self.W_rec = W_rec
		self.tau = tau
		if W_readout is None:
			self.W_readout = torch.zeros(N, dtype=torch.float32).to(device)
		else:
			self.W_readout = W_readout
		if W_feedback is None:
			self.W_feedback = torch.zeros(N, dtype=torch.float32).to(device)
		else:
			self.W_feedback = W_feedback
		self.activation = activation
		self.training_toggle=False
		self.P = None
		self.batch_size = 0
		self.learning_rule = None
		# Containers for monitors
		self.x_record = None
		self.r_record = None
		self.z_record = None
		self.W_readout_record = None
		self.dt_sampling = None

	def enable_training(self, learning_rule:callable=RLS_inplace, batch_size:int=1, alpha:torch.float32=1):
		self.learning_rule = learning_rule
		self.batch_size=batch_size
		self.P = (1.0/alpha)*torch.eye(self.N).to(device)
		self.training_toggle=True
		return self
	
	def feed_data(self, inputs, targets):
		self.inputs = inputs
		self.targets = targets
		return self

	def run(self, duration:torch.float32, dt:torch.float32=0.5):
		if self.dt_sampling is not None:
			for var in ('x', 'r', 'z', 'W_readout'):
				if self.__dict__[var+'_record'] is not None:
					container_buff = torch.zeros([int(duration/self.dt_sampling),*self.__dict__[var].shape,]).to(device)
					self.__dict__[var+'_record'] = torch.cat(
						(self.__dict__[var+'_record'], container_buff), dim=0
					) 
			ratio = self.dt_sampling/dt
			counter = 1

		for i in np.arange(duration/dt).astype(int):
			self.x += dt/self.tau * (-self.x + self.W_rec @ self.r + self.W_feedback*self.z + self.inputs[:,i])
			self.r = self.activation(self.x)
			self.z = self.W_readout @ self.r
			if self.training_toggle and ((i+1) % self.batch_size) == 0: # and training_flag[ti]:
				# update the error for the linear readout
				e = self.z-self.targets[i]
				# inplace update
				RLS_inplace(w=self.W_readout, error=e, P=self.P, r=self.r)
				# LMS_inplace(w=self.W_readout, error=e, eta=1e-3, r=self.r)
			# recording variables
			if self.dt_sampling is not None:
				for var in ('x', 'r', 'z', 'W_readout'):
					if (i+1) % ratio == 0:
						if self.__dict__[var+'_record'] is not None:
							self.__dict__[var+'_record'][counter] = self.__dict__[var]
				counter += 1


	def set_recorder(self, var_kw:str, dt:float) -> None:
		if var_kw not in ('x', 'r', 'z', 'W_readout'):
			raise AttributeError(f'[{var_kw:s}] is not recordable.')
		self.__dict__[var_kw+'_record'] = torch.unsqueeze(self.__dict__[var_kw], dim=0).to(device)
		self.dt_sampling = dt


# class Monitor:
# 	def __init__(self, net:RNN, ) -> None:
# 		pass


if torch.cuda.is_available(): 	   # device agnostic code
    device = torch.device('cuda')  # and modularity
else:                              #
    device = torch.device('cpu')   #

# device = torch.device('cpu')

param = {}
torch.cuda.synchronize()
t0 = time.time()
N = param['N'] = 500
n_input = param['n_input'] = 1
N_readout  = param['N_readout '] = 1
p = param['p'] = 1
g = param['g'] = 1.2		# g greater than 1 leads to chaotic networks.
alpha = param['alpha'] = 10
dt = param['dt'] = 1
tau = param['tau'] = 100 #ms
learn_every = param['learn_every'] = 10

np.random.seed(0)
# sparse random matrix M
rand_pick = (np.random.rand(N,N)<p).astype(int)
scale = g/np.sqrt(p*N)
W_rec = param['W_rec'] = rand_pick * scale * np.random.randn(N,N)
W_rec = param['W_rec'] = torch.Tensor(W_rec).to(device)

W_readout = param['W_readout'] = torch.zeros(N).to(device)
W_feedback = param['W_feedback'] = torch.Tensor(2.0*(np.random.rand(N)-0.5)).to(device)

# generate connectivity matrix for control inputs
J_GI = torch.Tensor(2.0*(np.random.rand(N, 1)-0.5))

# generate network
net = RNN(**param)

# print simulation setting
for key, val in param.items():
	if type(val) != torch.Tensor:
		print(f"{key:s} : ", val, sep=' ')
		

# construct input and target
# generate targets:
n_targets = param['n_targets'] = 3
seed = param['seed'] = 10
np.random.seed(seed)
I_range = [1,5]
I_duration=500 #ms
I_delay_range=[500, 6000] #ms
input = np.array([0])
target = np.array([0])
for i_strength, i_delay in zip(
	(1.8, 3.0, 4.9, ),
	(6000, 6000, 6000, ),
	# np.random.rand(n_targets)*(I_range[1]-I_range[0])+I_range[0],
	# np.random.rand(n_targets)*(I_delay_range[1]-I_delay_range[0])+I_delay_range[0],
):
	input = np.append(input, np.ones(int(I_duration/dt))*i_strength)
	input = np.append(input, np.zeros(int(i_delay/dt)))
	target = np.append(target, np.ones(int((I_duration+i_delay)/dt))*i_strength)

net.feed_data(torch.Tensor(np.outer(J_GI.flatten(), input)).to(device), torch.Tensor(target).to(device))

simtime = np.arange(len(target)) * dt

net.enable_training(batch_size=learn_every, alpha=alpha)
net.set_recorder('z', 1)
# net.set_recorder('x', 1)
# net.set_recorder('r', 1)
net.set_recorder('W_readout', 1)

# # create container for buffer variables
torch.cuda.synchronize()
print(f'matrix init takes {time.time()-t0:.3f} s')

torch.cuda.synchronize()
t0 = time.time()
net.run(target.shape[0],dt)
torch.cuda.synchronize()
print(f'evolve dynamics takes {time.time()-t0:.3f} s')

# # save trained model parameters
# wt_cpu = np.array(net.W_readout_record.cpu(), dtype=float)
# zt_cpu = np.array(net.z_record.cpu(), dtype=float)
# with open(f'memory-task_{n_targets:d}_cfg1.json', 'w') as write_file:
#     json.dump(param, write_file, indent=2)
# np.savez(f'memory-task_{n_targets:d}_net_hyper_pm1.npz', Jgg=W_rec.cpu(), Jgi=J_GI.cpu(), I = input, wf = W_feedback.cpu())

# np.savez(f'memory-task_{n_targets:d}_training_dynamics1.npz', ft=target, wt=wt_cpu, zt = zt_cpu)#, xt=xt.cpu())

# print training error
error_avg = torch.mean(torch.abs(net.z_record[1:].cpu().flatten()-torch.Tensor(target)))
print(f'Training MAE:  {error_avg:3e}')    

# save final frame as figure
fig2, ax2 = plt.subplots(2,1,figsize=(20,10))
ax2[0].plot(simtime, input, color='red', label='input')
ax2[0].plot(simtime, target, color='navy', label='f')
ax2[0].plot(simtime, net.z_record[1:].cpu().flatten(), color='orange', label='z', ls='--')
ax2[0].set_title('Training')
ax2[0].set_xlabel('Time(ms)')
ax2[0].set_ylabel(r'$f$ and $z$')
ax2[0].legend()

ax2[1].plot(simtime, torch.sqrt(torch.sum(net.W_readout_record[1:]**2,axis=1)).cpu(), label='|w|')
ax2[1].set_xlabel('Time(ms)')
ax2[1].set_ylabel(r'|$w$|')
ax2[1].legend()
plt.tight_layout()

plt.savefig(f'FORCE_Type_A_Memory_Task_{n_targets:d}Training_unified_test.png')