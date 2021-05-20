# FORCE_external_feedback_loop_multi-task_test.py
#	
# Test the trained multi-task network
#
# MATLAB vertion written by David Sussillo
# Modified by Kai Chen
#
# Benchmarks:	
# 	10 pattern case
# 	pure numpy version: 	196.64 s
#	PyTorch CUDA version: 	 17.60 s
#	PyTorch CPU version: 	 29.19 s


# %%
import numpy as np
import matplotlib.pyplot as plt
n_targets=8
q_value_FORCE = np.load(f'memory-task_{n_targets:d}_qvalue_final_result_FORCE.npz', allow_pickle=True)
q_value_LMS   = np.load(f'memory-task_{n_targets:d}_qvalue_final_result_LMS.npz', allow_pickle=True)
# %%
# save final frame as figure
I_range = [1,5]
i_strengthes = np.linspace(I_range[0],I_range[1],50)
fig2, ax2 = plt.subplots(1,1,)
ax2.semilogy(i_strengthes, q_value_FORCE['q'], color='navy', label='FORCE')
ax2.semilogy(i_strengthes, q_value_LMS['q'], color='orange', label='LMS')
ax2.set_title(f'Test')
ax2.set_xlabel('Target Value')
ax2.set_ylabel('q value')
ax2.legend()

plt.tight_layout()
fig2.savefig(f'FORCE_Type_A_Memory_Task_{n_targets:d}_qvalue_final_comp.png')
# %%
