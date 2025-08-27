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
import time
import torch
import json

if torch.cuda.is_available(): 	   # device agnostic code
    device = torch.device('cuda')  # and modularity
else:                              #
    device = torch.device('cpu')   #

mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['font.size'] = 12
mpl.rcParams['font.weight'] = 'bold'

torch.cuda.synchronize()
t0 = time.time()
with open('net_1_cfg.json', 'r') as read_file:
	param = json.load(read_file)
N = param['N']
p = param['p']
g = param['g']	# g greater than 1 leads to chaotic networks.
alpha = param['alpha']
nsecs_test = 2000
dt = param['dt']
learn_every = param['learn_every']

# load hyper parameters, i.e. network structure
# sparse matrix M
hyper_param = np.load('net_1_hyper.npz')
M = torch.Tensor(hyper_param['Jgg']).to(device)
wf = torch.Tensor(hyper_param['wf']).to(device)

n_rec2out = param['n_rec2out']
dw = torch.zeros(n_rec2out).to(device)

# print simulation setting
print('\tN: %d' % N)
print('\tg: %.3f' % g)
print('\tp: %.3f' % p)
print('\tn_rec2out: %d'% n_rec2out)
print('\talpha: %.3f' % alpha)
print('\tnsecs for test: %d' % nsecs_test)
print('\tlearn_every: %d' % learn_every)

from gen_patterns import gen_target, gen_sequential
amps = 1.3 / np.array([6.0, 2.0, 1.0, 3.0])

torch.manual_seed(0)
x0 = 0.5*torch.randn(N).to(device)
z0 = 0.5*torch.randn(1).to(device)

torch.cuda.synchronize()
print(f'matrix init takes {time.time()-t0:.3f} s')

torch.cuda.synchronize()
t0 = time.time()
period = 120
n_patterns = 1
test_MAE = np.zeros((n_patterns, 30))
wo_len = np.zeros_like(test_MAE)
for j in np.arange(test_MAE.shape[1]):
	simtime = np.arange(0, period*(j+1)+nsecs_test,dt)
	simtime_len = len(simtime)
	simtime_train_len = int(simtime_len/(period*(j+1)+nsecs_test)*(period*(j+1)))

	# fts = gen_sequential(num=n_patterns, time=simtime)
	fts = [gen_target(amps, simtime)]
	for i, ft in enumerate(fts):
		ft_cuda = torch.Tensor(ft).to(device)

		wt = torch.zeros((simtime_len,n_rec2out)).to(device)
		zt = torch.zeros(simtime_len).to(device)

		x = x0.clone()
		r = torch.tanh(x)
		z = z0.clone()
		wo = torch.zeros(n_rec2out).to(device)

		P = (1.0/alpha)*torch.eye(n_rec2out).to(device)
		for ti in np.arange(len(simtime)):
			# sim, so x(t) and r(t) are created.
			x += dt * (-x + M @ r + wf*z)
			r = torch.tanh(x)
			z = wo @ r
			
			if ((ti+1) % learn_every) == 0 and ti < simtime_train_len:
				# update inverse correlation matrix
				k = P @ r
				rPr = r @ k
				c = 1.0/(1.0 + rPr)
				P -= c * torch.outer(k,k)
				
				# update the error for the linear readout
				e = z-ft_cuda[ti]
				
				# update the output weights
				dw = -e*k*c	
				wo += dw
			
			# Store the output of the system.
			zt[ti] = z
			# wt[ti,:] = wo
		wo_len[i, j] = torch.sqrt(wo@wo).cpu()

		# calculate test error
		error_avg = torch.sum(torch.abs(zt[simtime_train_len:]-ft_cuda[simtime_train_len:]))/(simtime_len - simtime_train_len)
		test_MAE[i,j] = error_avg.cpu()

torch.cuda.synchronize()
print(f'evolve dynamics takes {time.time()-t0:.3f} s')
# save outputs
# ------------
# ft_cpu = np.array(ft.cpu(), dtype=float)
# wt_cpu = np.array(wt.cpu(), dtype=float)
# zt_cpu = np.array(zt.cpu(), dtype=float)
# with open(f'net_1_cfg.json', 'w') as write_file:
#     json.dump(param, write_file, indent=2)
np.savez('tmp_data.npz', test_MAE=test_MAE, wo_len=wo_len)

# from gen_patterns import gen_target, gen_sequential
# period = 120
# plot_data = np.load('tmp_data.npz')
# test_MAE = plot_data['test_MAE']
# wo_len = plot_data['wo_len']

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from gen_patterns import make_ax_style
fts = gen_sequential(time=np.arange(0, period, 0.1))
# save final frame as figure
colors = ['orange', 'navy']
labels = ['Test MAE', r'$|w|$']
fig, axes = plt.subplots(5,5,figsize=(25,25))
axes = axes.flatten()
plt.tight_layout()
for idx in range(test_MAE.shape[0]):
	ax = axes[idx]
	line1 = ax.plot(np.arange(test_MAE.shape[1])+1, test_MAE[idx], color=colors[0])[0]
	# ax.set_title('Test MAE v.s. Training Epoches')
	# ax.set_xlabel('Training Epoches (# of period)')
	ax.yaxis.label.set_color(colors[0])
	[i.set_color(colors[0]) for i in ax.yaxis.get_ticklabels()]
	# ax.set_ylabel('Test MAE')
	ax2 = ax.twinx()
	line2=ax2.plot(np.arange(test_MAE.shape[1])+1, wo_len[idx], color=colors[1])[0]
	# ax2.set_ylabel(r'Length of $|w|$', rotation=90)
	ax2.yaxis.label.set_color(colors[1])
	[i.set_color(colors[1]) for i in ax2.yaxis.get_ticklabels()]
	plt.legend((line1, line2), labels, loc=2)

	inset_ax = inset_axes(ax,
						width="30%",
						height="30%",
						loc=7)
	inset_ax.plot(fts[idx], 'springgreen')
	make_ax_style(inset_ax)
	inset_ax.set_frame_on(True)
	inset_ax.set_title('Target Pattern')
plt.savefig('Figure_testMAE-epoch.png')