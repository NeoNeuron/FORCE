import numpy as np
import matplotlib.pyplot as plt
# import time
import torch
from gen_patterns import gen_sequential

def scan_corrcoef(x:torch.Tensor, y:torch.Tensor, arg:bool=False)->float:
    # define corrcoef for cuda
    def corrcoef(x:torch.Tensor, y:torch.Tensor)->torch.Tensor:
        cov = (x*y).mean() - x.mean()*y.mean()
        if cov.abs() <= 1e-5:
            return cov
        else:
            return cov/x.std()/y.std()
    # scan across time delay
    buffer = np.zeros(int(120/0.1))
    print(buffer.shape)
    for i in np.arange(buffer.shape[0])+1:
        buffer[i-1] = corrcoef(x[i:], y[:-i]).cpu()	
    val_max = buffer[~np.isnan(buffer)].max()
    if arg:
        arg_max = np.argwhere(buffer==val_max)[0,0]
        return val_max, arg_max
    else:
        return val_max

device = torch.device('cuda')
output_data = np.load('dbg_tmp_data.npy')

dt = 0.1
len = output_data[0].shape[0]
time = np.arange(len)*dt

n = 10

target_pattern = gen_sequential(n, time)
target_pattern_cuda = [torch.Tensor(item).to(device) for item in target_pattern]
output_data_cuda = [torch.Tensor(item).to(device) for item in output_data]

fig, ax = plt.subplots(10,1, figsize=(10,20))
for i, (x,y) in enumerate(zip(target_pattern, output_data)):
    ax[i].plot(x[len//2:], 'g') 
    ax[i].plot(y[len//2:], 'r') 

plt.tight_layout()
plt.savefig('gdb_tmp.png')

for x,y in zip(target_pattern_cuda, output_data_cuda):
    print(scan_corrcoef(x[len//2:], y[len//2:]))
    # cov = (x[len//2:]* y[len//2:]).mean()- x[len//2:].mean()*y[len//2:].mean()
    # if cov.abs() <= 1e-5:
    #     print(cov)
    # else:
    #     print(cov/x[len//2:].std()/y[len//2:].std())