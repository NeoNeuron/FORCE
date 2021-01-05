# training patterns
#
# Generate training patterns

import numpy as np 
import itertools
import matplotlib.pyplot as plt 

period = 120
n_options = 24

def gen_target(amps:np.ndarray, time:np.ndarray)->np.ndarray:
    freqs = 1/60 * np.array([1.0, 2.0, 3.0, 4.0])
    ft = np.zeros_like(time)
    for amp, freq in zip(amps, freqs):
        ft += amp*np.sin(np.pi*freq*time)
    ft = ft/1.5
    return ft

def gen_sequential_options(num:int=None)->list:
    option_pool = list(itertools.permutations([1.0,2.0,3.0,6.0],4))
    if num is None:
        return option_pool
    else:
        return option_pool[:num]

def gen_random_options(num:int=None, seed=None)->list:
    option_pool = list(itertools.permutations([1.0,2.0,3.0,6.0],4))
    rg = np.random.Generator(np.random.MT19937(seed))
    if num is None:
        return rg.choice(option_pool, len(option_pool), replace=False)
    else:
        return rg.choice(option_pool, num, replace=False)

def gen_sequential(num:int=None, time:np.ndarray=np.arange(0,period,0.1))->list:
    '''Generate a list of target functions with sequential order.
    :param num: number of targets. Default: 24
    :param time: time array of target functions. Default: one period of time (120)
    :param seed: seed of random Generator. Generator is implemented as MT19937.
    :return: list of target patterns.

    '''
    return [gen_target(1.3/np.array(opt),time) for opt in gen_sequential_options(num)]

def gen_random(num:int=None, time:np.ndarray=np.arange(0,period,0.1), seed:int=None)->list:
    '''Generate a list of target functions with random order.
    :param num: number of targets. Default: 24
    :param time: time array of target functions. Default: one period of time (120)
    :param seed: seed of random Generator. Generator is implemented as MT19937.
    :return: list of target patterns.

    '''
    return [gen_target(1.3/np.array(opt),time) for opt in gen_random_options(num, seed)]

def make_ax_style(ax: plt.Axes)->plt.Axes:
    # make axis invisible
    ax.set_frame_on(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    return ax

# # shifting order
# order = np.arange(n_options, dtype=int)
# rg.shuffle(order)
# options = [options[idx] for idx in order]

if __name__ == '__main__':
    dt = 0.1
    time = np.arange(0, period, dt)
    fig, ax = plt.subplots(5,5, figsize=(25,25))
    ax = ax.flatten()
    fts = gen_sequential(time=time)
    for i in range(len(fts)):
        ax[i].plot(time, fts[i], lw=3, color='navy')
        ax[i].set_title(f'Patten {i+1:d}', fontsize=30)
    # Remove frame edges
    for axi in ax:
        make_ax_style(axi)

    plt.tight_layout()
    plt.savefig('patterns.png')