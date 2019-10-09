import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from .data import _detransform

def plot_spectrum(means, scales, weights=None, nyquist=None, titles=None, show=True, filename=None, title=None):
    # sns.set(font_scale=2)
    # sns.axes_style("darkgrid")
    # sns.set_style("whitegrid")
    
    if means.ndim == 2:
        means = np.expand_dims(means, axis=2)
    if scales.ndim == 2:
        scales = np.expand_dims(scales, axis=2)
    if isinstance(weights, np.ndarray) and weights.ndim == 1:
        weights = np.expand_dims(weights, axis=1)
    if isinstance(nyquist, np.ndarray) and nyquist.ndim == 1:
        nyquist = np.expand_dims(nyquist, axis=1)

    if means.ndim != 3:
        raise Exception('means and scales must have shape (mixtures,input_dims,output_dims)')
    if means.shape != scales.shape:
        raise Exception('means and scales must have the same shape (mixtures,input_dims,output_dims)')

    mixtures = means.shape[0]
    input_dims = means.shape[1]
    output_dims = means.shape[2]

    if isinstance(weights, np.ndarray) and (weights.ndim != 2 or weights.shape[0] != mixtures or weights.shape[1] != output_dims):
        raise Exception('weights must have shape (mixtures,output_dims)')
    elif not isinstance(weights, np.ndarray):
        weights = np.ones((mixtures,output_dims))
    if isinstance(nyquist, np.ndarray) and (nyquist.ndim != 2 or nyquist.shape[0] != input_dims or nyquist.shape[1] != output_dims):
        raise Exception('nyquist must have shape (input_dims,output_dims)')

    fig, axes = plt.subplots(output_dims, input_dims, figsize=(20, output_dims*5), sharey=False, constrained_layout=True, squeeze=False)
    if title != None:
        fig.suptitle(title, fontsize=36)
    
    for channel in range(output_dims):
        for i in range(input_dims):
            x_low = max(0.0, norm.ppf(0.01, loc=means[:,i,channel], scale=scales[:,i,channel]).min())
            x_high = norm.ppf(0.99, loc=means[:,i,channel], scale=scales[:,i,channel]).max()
            if isinstance(nyquist, np.ndarray):
                x_high = min(x_high, nyquist[i,channel])

            x = np.linspace(x_low, x_high, 1000)
            psd = np.zeros(x.shape)

            for q in range(mixtures):
                single_psd = weights[q,channel] * norm.pdf(x, loc=means[q,i,channel], scale=scales[q,i,channel])
                #single_psd = np.log(single_psd+1)
                axes[channel,i].plot(x, single_psd, '--', c='r', zorder=2)
                psd += single_psd
           
            axes[channel,i].plot(x, psd, 'k-', zorder=1)
            axes[channel,i].set_yticks([])
            axes[channel,i].set_ylim(0, None)
            if titles != None:
                axes[channel,i].set_title(titles[channel])

    axes[output_dims-1,i].set_xlabel('Frequency')

    if filename != None:
        plt.savefig(filename+'.pdf', dpi=300)
    if show:
        plt.show()

    return fig, axes


def plot_prediction(model, grid=None, figsize=(12, 8), ylims=None, names=None, title=''):

    """
    Plot training points, all data and prediction for all channels.

    Args:
        Model (mogptk.Model object): Model to use.
        grid (tuple) : Tuple with the 2 dimensions of the grid.
        figsize(tuple): Figure size, default to (12, 8).
        ylims(list): List of tuples with limits for Y axis for
            each channel.
        Names(list): List of the names of each title.
        title(str): Title of the plot.
    """
    # get data
    x_train = [c.X[c.mask] for c in model.data]
    y_train = [_detransform(c.transformations, c.X[c.mask], c.Y[c.mask]) for c in model.data]
    x_all = [c.X for c in model.data]
    y_all = [_detransform(c.transformations, c.X, c.Y) for c in model.data]
    x_pred = [c.X for c in model.data]

    n_dim = model.get_output_dims()

    if (grid[0] * grid[1]) < n_dim:
        raise Exception('Grid not big enough for all channels')

    if grid is None:
        grid = (np.ceil(n_dim/2), 2)

    # predict with model
    mean_pred, lower_ci, upper_ci = model.predict(x_pred)

    # create plot
    f, axarr = plt.subplots(grid[0], grid[1], sharex=True, figsize=figsize)

    axarr = axarr.reshape(-1)

    # plot
    for i in range(n_dim):
        axarr[i].plot(x_train[i][:, 0], y_train[i], '.k', label='Train', ms=4)
        axarr[i].plot(x_all[i][:, 0], y_all[i], '--', label='Test', c='gray',lw=1.4, zorder=5)
        
        axarr[i].plot(x_pred[i][:, 0], mean_pred[i], label='Post.Mean', c=sns.color_palette()[i%10], zorder=1)
        axarr[i].fill_between(x_pred[i][:, 0].reshape(-1),
                              lower_ci[i],
                              upper_ci[i],
                              label='95% c.i',
                              color=sns.color_palette()[i%10],
                              alpha=0.4)
        
        # axarr[i].legend(ncol=4, loc='upper center', fontsize=8)
        axarr[i].set_xlim(x_all[i][0]-1, x_all[i][-1])

        # set channels name
        if names is not None:
            axarr[i].set_title(names[i])
        else:
            axarr[i].set_title('Channel' + str(i))

        # set y lims
        if ylims is not None:
            axarr[i].set_ylim(ylims[i]) 
        
    plt.suptitle(title, y=1.02)
    plt.tight_layout()

    return f, axarr

