import numpy as np
from scipy.sparse import csr_matrix
import time
import torch

if torch.cuda.is_available(): 	   # device agnostic code
    device = torch.device('cuda')  # and modularity
else:                              #
    device = torch.device('cpu')   #

# device = torch.device('cpu')   #
N = 1000
p = 0.1
g = 1.5		# g greater than 1 leads to chaotic networks.
alpha = 1.0
nsecs = 4000
nsecs_train = 2000
dt = 0.1
learn_every = 2

torch.cuda.synchronize()
t0 = time.time()
# sparse matrix M
M = np.random.rand(N,N)
M = (M<p).astype(int)
scale = 1.0/np.sqrt(p*N)
M = M * g*scale * np.random.randn(N,N)
M_tensor = torch.Tensor(M).to(device)
# M_spa = csr_matrix(M)

nRec2Out = N
wo = torch.zeros(nRec2Out).to(device)
wf = 2.0*(torch.rand(N)-0.5).to(device)

# wo_len = np.zeros(simtime_len)
# zt = np.zeros(simtime_len)
x0 = 0.5*torch.randn(N).to(device)
z0 = 0.5*torch.randn(1).to(device)

x = x0 
z = z0
r = torch.tanh(x)

simtime = np.arange(0,nsecs,dt)
simtime_len = len(simtime)
simtime_train_len = int(simtime_len/nsecs*nsecs_train)

amps = 1.3 / np.array([1.0, 2.0, 6.0, 3.0])
freqs = 1/60 * np.array([1.0, 2.0, 3.0, 4.0])
ft = np.zeros_like(simtime)
for amp, freq in zip(amps, freqs):
	ft += amp*np.sin(np.pi*freq*simtime)
ft = ft/1.5
ft = torch.Tensor(ft).to(device)

P = (1.0/alpha)*torch.eye(nRec2Out).to(device)
torch.cuda.synchronize()
print(f'matrix init takes {time.time()-t0:.3f} s')

torch.cuda.synchronize()
t0 = time.time()
# test one loop
for ti in range(10000):
    x += dt * (-x + M_tensor @ r + wf*z)
    r = torch.tanh(x).to(device)
    z = wo @ r
    if ((ti+1) % learn_every) == 0 and ti < simtime_train_len:
        # update inverse correlation matrix
        k = P @ r
        rPr = r @ k
        c = 1.0/(1.0 + rPr)
        P -= c * torch.outer(k,k)

        # update the error for the linear readout
        e = z-ft[ti]

        # update the output weights
        dw = -e*k*c	
        wo += dw
torch.cuda.synchronize()
print(f'matrix calculation takes {time.time()-t0:.3f} s')

# print(x)
# print(z)
# print(x.to('cpu'))
# print(z.to('cpu'))
