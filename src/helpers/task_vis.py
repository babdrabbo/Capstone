import numpy as np
from helpers.task import Task
from matplotlib import colors
from matplotlib import pyplot as plt

cmap = colors.ListedColormap(
        ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00', 
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

norm = colors.Normalize(vmin=0, vmax=9)


def initAxis(ax, grid, title):
    offSet = -.5 # Set offset value (for bug)
    ax.grid(alpha=0.5) # Show grid
    ax.set_xticklabels([]) # Remove x labels
    ax.set_yticklabels([]) # Remove y labels
    shape = np.asarray(grid).shape # Get shape 
    ax.set_xticks(np.arange(offSet, shape[1]))
    ax.set_yticks(np.arange(offSet, shape[0]))
    ax.set_title(title + ' ' + str(shape))
    # ax.set_xlabel(shape[1]) # Set x label
    # ylbl = ax.set_ylabel(shape[0]) # Set y label
    # ylbl.set_rotation(0) # Reset y label rotation
    ax.tick_params(length=0) # Set tick size to zero
    ax.imshow(grid, cmap=cmap, norm=norm) # Show plot
    
    
def plot_grid(grid, title=''):
    _, axs = plt.subplots(1, 1)
    initAxis(axs, grid, title)
    plt.tight_layout(pad=3.0)
    plt.show()

def plot_grid_pairs(pairs, w=4):
    n = len(pairs)
    fig, axs = plt.subplots(n, 2, figsize=(2*w,n*w))
    for i, pair in enumerate(pairs):
        axa = axs[i, 0] if n > 1 else axs[0]
        axb = axs[i, 1] if n > 1 else axs[1]
        initAxis(axa, pair[0], f'in {i}')
        initAxis(axb, pair[1], f'out {i}')
    
    plt.tight_layout(pad=3.0)
    plt.show()


def plot_task_sample(task:Task, ex_idx=0, test_idx=0):

    _, axs = plt.subplots(1, 4, figsize=(15,10))
    print(f'type(axs): {type(axs)}')

    # Train input/output
    trnIn, trnOut = task.get_examples()[ex_idx]
    initAxis(axs[0], trnIn, f'Example {ex_idx} Input')
    initAxis(axs[1], trnOut, f'Example {ex_idx} Output')

    # Test input/output
    tstIn, tstOut = list(zip(task.get_tests(), task.get_solutions()))[test_idx]
    initAxis(axs[2], tstIn, f'Test {test_idx} Input')
    initAxis(axs[3], tstOut, f'Test {test_idx} Output')

    plt.tight_layout(pad=3.0)
    plt.show()
