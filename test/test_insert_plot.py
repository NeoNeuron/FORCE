# test insert plot
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from gen_patterns import make_ax_style

mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['font.size'] = 18
mpl.rcParams['font.weight'] = 'bold'

plot_data = np.load('tmp_data.npz')
test_MAE = plot_data['test_MAE']
wo_len = plot_data['wo_len']

# save final frame as figure
colors = ['orange', 'navy']
lines = []
labels = ['Test MAE', r'$|w|$']
fig, ax = plt.subplots(1,1,figsize=(10,8))
line1 = ax.plot(np.arange(1, test_MAE.shape[0]+1), test_MAE, color=colors[0])[0]
ax.set_title('Test MAE v.s. Training Epoches')
ax.set_xlabel('Training Epoches (# of period)')
ax.yaxis.label.set_color(colors[0])
[i.set_color(colors[0]) for i in ax.yaxis.get_ticklabels()]
ax.set_ylabel('Test MAE')
# ax.legend()
ax2 = ax.twinx()
line2=ax2.plot(np.arange(1, test_MAE.shape[0]+1), wo_len, color=colors[1])[0]
ax2.set_ylabel(r'Length of $|w|$', rotation=90)
ax2.yaxis.label.set_color(colors[1])
[i.set_color(colors[1]) for i in ax2.yaxis.get_ticklabels()]
plt.legend((line1, line2), labels)
plt.tight_layout()
# insert plot
from gen_patterns import gen_target
amps = 1.3 / np.array([1.0, 2.0, 6.0, 3.0])
ft = gen_target(amps, np.arange(0,120,0.1))
inset_axes = inset_axes(ax,
                    width="30%", # width = 30% of parent_bbox
                    height=2., # height : 1 inch
                    loc=7)
inset_axes.plot(ft, 'springgreen')
make_ax_style(inset_axes)
inset_axes.set_frame_on(True)
inset_axes.set_title('Target Pattern')
plt.savefig('Figure_testMAE-epoch.png')
