# %%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# %%
n_targets=10
data = np.load(f'memory-task_{n_targets:d}_training_dynamics.npz', allow_pickle=True)
ft=data['ft']
wt=data['wt']
zt=data['zt']
xt=data['xt']
# %%
pca = PCA(n_components=2)
pca.fit(xt)

# %%
w_reduce = pca.transform(wt)
# %%
my_color = plt.cm.jet_r(np.arange(w_reduce.shape[0])/w_reduce.shape[0])
input_flag = np.nonzero(np.diff(ft))[0]
input_flag = np.append(input_flag[1:], -1)

fig = plt.figure(figsize=(6,5))
gs1 = fig.add_gridspec(nrows=1, ncols=1, 
                      left=0.08, right=0.86, top=0.96, bottom=0.08, 
                      wspace=0., hspace=0.)
gs2 = fig.add_gridspec(nrows=1, ncols=1, 
                      left=0.89, right=0.92, top=0.96, bottom=0.08, 
                      wspace=0., hspace=0.)
ax = np.array([fig.add_subplot(i[0]) for i in [gs1, gs2]])
ax[0].scatter(w_reduce[:,0], w_reduce[:,1], c=my_color, s=2)
ax[0].scatter(w_reduce[input_flag,0], w_reduce[input_flag,1], color=(0,0,0,0), edgecolors='k', s=50, marker='s')
ax[0].set_xlabel('PC1', fontsize=20)
ax[0].set_ylabel('PC2', fontsize=20)
ax[0].set_xticks([])
ax[0].set_yticks([])
# %%
print(ft[input_flag])

norm = mpl.colors.Normalize(vmin=0, vmax=10)

fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=plt.cm.jet_r),
             cax=ax[1], orientation='vertical', label='Order of inputs')


# %%
plt.savefig(f'w2pca_{n_targets:d}.png')
