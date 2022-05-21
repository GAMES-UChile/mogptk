import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from .serie import TransformLinear

def plot_spectrum(means, scales, dataset=None, weights=None, nyquist=None, noises=None, log=False, titles=None, show=True, filename=None, title=None):
    """
    Plot spectral Gaussians of given means, scales and weights.
    """
    if means.ndim == 2:
        means = np.expand_dims(means, axis=2)
    if scales.ndim == 2:
        scales = np.expand_dims(scales, axis=2)
    if isinstance(weights, np.ndarray) and weights.ndim == 1:
        weights = np.expand_dims(weights, axis=1)
    if isinstance(nyquist, np.ndarray) and nyquist.ndim == 1:
        nyquist = np.expand_dims(nyquist, axis=1)

    if means.ndim != 3:
        raise ValueError('means and scales must have shape (mixtures,output_dims,input_dims)')
    if means.shape != scales.shape:
        raise ValueError('means and scales must have the same shape (mixtures,output_dims,input_dims)')
    if noises is not None and (noises.ndim != 1 or noises.shape[0] != means.shape[1]):
        raise ValueError('noises must have shape (output_dims,)')
    if dataset is not None and len(dataset) != means.shape[1]:
        raise ValueError('means and scales must have %d output dimensions' % len(dataset))

    mixtures = means.shape[0]
    output_dims = means.shape[1]
    input_dims = means.shape[2]

    if isinstance(weights, np.ndarray) and (weights.ndim != 2 or weights.shape[0] != mixtures or weights.shape[1] != output_dims):
        raise ValueError('weights must have shape (mixtures,output_dims)')
    elif not isinstance(weights, np.ndarray):
        weights = np.ones((mixtures,output_dims))
    if isinstance(nyquist, np.ndarray) and (nyquist.ndim != 2 or nyquist.shape[0] != output_dims or nyquist.shape[1] != input_dims):
        raise ValueError('nyquist must have shape (output_dims,input_dims)')

    h = 3.0*output_dims
    fig, axes = plt.subplots(output_dims, input_dims, figsize=(12,h), squeeze=False, constrained_layout=True)
    if title is not None:
        fig.suptitle(title, y=(h+0.8)/h, fontsize=18)
    
    for j in range(output_dims):
        for i in range(input_dims):
            if dataset is not None and len(dataset[j].X[i].transformers) == 1 and isinstance(dataset[j].X[i].transformers[0], TransformLinear):
                means[:,j,:] /= dataset[j].X[i].transformers[0].slope
                scales[:,j,:] /= dataset[j].X[i].transformers[0].slope

            x_low = max(0.0, norm.ppf(0.01, loc=means[:,j,i], scale=scales[:,j,i]).min())
            x_high = norm.ppf(0.99, loc=means[:,j,i], scale=scales[:,j,i]).max()

            if dataset is not None:
                X = dataset[j].X[:,i]
                if 1 < len(X):
                    idx = np.argsort(X)
                    X = X[idx]
                    dist = np.abs(X[1:]-X[:-1])
                    nyquist_data = 0.5 / np.average(dist)
                    x_low = 0.5 / np.abs(X[-1]-X[0])
                    x_high = nyquist_data
                dataset[j].plot_spectrum(ax=axes[j,i], method='ls', transformed=False, log=False)
            elif isinstance(nyquist, np.ndarray):
                x_high = min(x_high, nyquist[j,i])

            x = np.linspace(x_low, x_high, 1001)
            psd = np.zeros(x.shape)
            for q in range(mixtures):
                single_psd = weights[q,j] * norm.pdf(x, loc=means[q,j,i], scale=scales[q,j,i])
                #single_psd = np.log(single_psd+1)
                #axes[j,i].plot(x, single_psd, ls='--', c='k', zorder=2)
                axes[j,i].axvline(means[q,j,i], ymin=0.001, ymax=0.05, lw=3, color='C1')
                psd += single_psd
            if noises is not None:
                psd += noises[j]**2

            # normalize
            #psd /= psd.sum() * (x[1]-x[0])

            y_low = 0.0
            if log:
                x_low = max(x_low, 1e-8)
                y_low = 1e-8
           
            axes[j,i].plot(x, psd, ls='-', c='C0')
            axes[j,i].set_xlim(x_low, x_high)
            axes[j,i].set_ylim(y_low, None)
            axes[j,i].set_yticks([])
            if titles is not None:
                axes[j,i].set_title(titles[j])
            if log:
                axes[j,i].set_xscale('log')
                axes[j,i].set_yscale('log')

    axes[output_dims-1,i].set_xlabel('Frequency')

    legends = []
    if dataset is not None:
        legends.append(plt.Line2D([0], [0], ls='-', color='k', label='Data (LombScargle)'))
    legends.append(plt.Line2D([0], [0], ls='-', color='C0', label='Model'))
    #legends.append(plt.Line2D([0], [0], ls='--', color='k', label='Mixture'))
    legends.append(plt.Line2D([0], [0], ls='-', color='C1', label='Peak location'))
    fig.legend(handles=legends)#, loc="upper center", bbox_to_anchor=(0.5,(h+0.4)/h), ncol=3)

    if filename is not None:
        plt.savefig(filename+'.pdf', dpi=300)
    if show:
        plt.show()
    return fig, axes

