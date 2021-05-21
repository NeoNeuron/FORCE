# %%
import numpy as np
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
pca = PCA(n_components=10)
pca.fit(xt)

# print(pca.explained_variance_ratio_)

# print(pca.singular_values_)
# plt.semilogy(pca.singular_values_)

# %%
# plt.semilogy(pca.singular_values_*6e3)
# plt.axhline(1/20)
# %%
w_reduce = pca.transform(wt)
# %%
my_color = plt.cm.rainbow(np.arange(w_reduce.shape[0])/w_reduce.shape[0])
input_flag = np.nonzero(np.diff(ft))[0]
input_flag = np.append(input_flag[1:], -1)

plt.scatter(w_reduce[:,0], w_reduce[:,1], c=my_color, s=2)
plt.scatter(w_reduce[input_flag,0], w_reduce[input_flag,1], c='k', s=4)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.xticks([])
plt.yticks([])
# %%
print(ft[input_flag])
# %%
plt.savefig(f'w2pca_{n_targets:d}.png')
