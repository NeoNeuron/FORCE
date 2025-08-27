#%%
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
from FORCE import RNN

if torch.cuda.is_available(): 	   # device agnostic code
    device = torch.device('cuda')  # and modularity
else:                              #
    device = torch.device('cpu')   #

# device = torch.device('cpu')

param = {}
torch.cuda.synchronize()
t0 = time.time()
N = param['N'] = 1000
n_input = param['n_input'] = 1
N_readout  = param['N_readout '] = 1
p = param['p'] = 1
g = param['g'] = 1.0		# g greater than 1 leads to chaotic networks.
alpha = param['alpha'] = 0.1
dt = param['dt'] = 1
tau = param['tau'] = 1 #ms
learn_every = param['learn_every'] = 1

np.random.seed(0)
# sparse random matrix M
rand_pick = (np.random.rand(N,N)<p).astype(int)
scale = g/np.sqrt(p*N)
W_rec = param['W_rec'] = rand_pick * scale * np.random.randn(N,N)
W_rec = param['W_rec'] = torch.Tensor(W_rec).to(device)

W_readout = param['W_readout'] = torch.zeros(N).to(device)
# W_feedback = param['W_feedback'] = torch.Tensor(2.0*(np.random.rand(N)-0.5)).to(device)
W_feedback = param['W_feedback'] = torch.Tensor(np.zeros(N)).to(device)

# generate connectivity matrix for control inputs
J_GI = torch.Tensor(2.0*(np.random.rand(N, 1)-0.5))

# generate network
net = RNN(**param)
net.init_training(batch_size=learn_every, alpha=alpha)

# print simulation setting
for key, val in param.items():
	if type(val) != torch.Tensor:
		print(f"{key:s} : ", val, sep=' ')
		
# set variable to be recorded
net.set_recorder('z', 1)
# net.set_recorder('x', 1)
# net.set_recorder('r', 1)
net.set_recorder('W_readout', 1)

# construct input and target
# generate targets:
seed = param['seed'] = 10
np.random.seed(seed)
I_duration=1000 #number of data points
input_raw = np.random.rand(int(2*I_duration))*2-1

n_tau = 10
train_error = np.zeros(n_tau)
test_error = np.zeros(n_tau)
for tau in np.arange(n_tau):
    if tau > 0:
        input = input_raw.copy()
        target = np.hstack((input[-tau:], input[:-tau]))
    else:
        input = input_raw.copy()
        target = input_raw.copy()

    # set variable to be recorded
    net.set_recorder('z', 1)
    # net.set_recorder('x', 1)
    # net.set_recorder('r', 1)
    net.set_recorder('W_readout', 1)
    net.init_training(batch_size=learn_every, alpha=alpha)

    # create container for buffer variables
    torch.cuda.synchronize()
    t0 = time.time()
    net.run(
        torch.Tensor(np.outer(J_GI.flatten(), input[:I_duration])).to(device),
        torch.Tensor(target[:I_duration]).to(device),dt)
    torch.cuda.synchronize()
    # print(f'evolve dynamics takes {time.time()-t0:.3f} s')

    # print training error
    train_error[tau] = np.abs(np.corrcoef(np.array(net.z_record.cpu().flatten())[200:], target[200:I_duration])[0,1])
    # train_error[tau] = torch.mean(torch.abs(net.z_record.cpu().flatten()[200:]-torch.Tensor(target[200:I_duration])))
    print(f'Training corr:  {train_error[tau]:3e}')

    # reset container for buffer variables
    net.set_recorder('z', 1)
    net.set_recorder('W_readout', 1)
    torch.cuda.synchronize()
    t0 = time.time()
    net.run(
        torch.Tensor(np.outer(J_GI.flatten(), input[I_duration:])).to(device),
        None, dt)
    torch.cuda.synchronize()
    # print(f'evolve test dynamics takes {time.time()-t0:.3f} s')

    # print training error
    test_error[tau] = np.abs(np.corrcoef(np.array(net.z_record.cpu().flatten()), target[I_duration:])[0,1])
    # test_error[tau] = torch.mean(torch.abs(net.z_record.cpu().flatten()-torch.Tensor(target[I_duration:])))
    print(f'Test corr:  {test_error[tau]:3e}')

plt.plot(np.arange(n_tau), train_error, '-o', color='r', label='Train Corr')
plt.plot(np.arange(n_tau), test_error, '-o', color='g', label='Test Corr')
plt.legend()

# save trained model parameters
# wt_cpu = np.array(net.W_readout_record.cpu(), dtype=float)
# zt_cpu = np.array(net.z_record.cpu(), dtype=float)
# with open(f'memory-task_{n_targets:d}_cfg1.json', 'w') as write_file:
#     json.dump(param, write_file, indent=2)
# np.savez(f'memory-task_{n_targets:d}_net_hyper_pm1.npz', Jgg=W_rec.cpu(), Jgi=J_GI.cpu(), I = input, wf = W_feedback.cpu())

# np.savez(f'memory-task_{n_targets:d}_training_dynamics1.npz', ft=target, wt=wt_cpu, zt = zt_cpu)#, xt=xt.cpu())
#%%

simtime = np.arange(len(target)/2) * dt
# save final frame as figure
fig2, ax2 = plt.subplots(2,1,figsize=(20,10))
ax2[0].plot(simtime, input[:I_duration], color='red', label='input')
ax2[0].plot(simtime, target[:I_duration], color='navy', label='target')
ax2[0].plot(simtime, net.z_record.cpu().flatten(), color='orange', label='z', ls='--')
ax2[0].set_title('Training')
ax2[0].set_xlabel('Time(ms)')
ax2[0].set_ylabel(r'$f$ and $z$')
ax2[0].legend()
ax2[0].set_xlim(0,100)

ax2[1].plot(simtime, torch.sqrt(torch.sum(net.W_readout_record**2,axis=1)).cpu(), label='|w|')
ax2[1].set_xlabel('Time(ms)')
ax2[1].set_ylabel(r'|$w$|')
ax2[1].legend()
plt.tight_layout()

#%%