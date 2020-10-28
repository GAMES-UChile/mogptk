import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def plot_spectrum(means, scales, weights=None, nyquist=None, titles=None, show=True, filename=None, title=None):
    if means.ndim == 2:
        means = np.expand_dims(means, axis=2)
    if scales.ndim == 2:
        scales = np.expand_dims(scales, axis=2)
    if isinstance(weights, np.ndarray) and weights.ndim == 1:
        weights = np.expand_dims(weights, axis=1)
    if isinstance(nyquist, np.ndarray) and nyquist.ndim == 1:
        nyquist = np.expand_dims(nyquist, axis=1)

    if means.ndim != 3:
        raise Exception('means and scales must have shape (mixtures,output_dims,input_dims)')
    if means.shape != scales.shape:
        raise Exception('means and scales must have the same shape (mixtures,output_dims,input_dims)')

    mixtures = means.shape[0]
    output_dims = means.shape[1]
    input_dims = means.shape[2]

    if isinstance(weights, np.ndarray) and (weights.ndim != 2 or weights.shape[0] != mixtures or weights.shape[1] != output_dims):
        raise Exception('weights must have shape (mixtures,output_dims)')
    elif not isinstance(weights, np.ndarray):
        weights = np.ones((mixtures,output_dims))
    if isinstance(nyquist, np.ndarray) and (nyquist.ndim != 2 or nyquist.shape[0] != output_dims or nyquist.shape[1] != input_dims):
        raise Exception('nyquist must have shape (output_dims,input_dims)')

    fig, axes = plt.subplots(output_dims, input_dims, figsize=(20, output_dims*5), squeeze=False)
    fig.set_tight_layout(True)
    if title is not None:
        fig.suptitle(title, fontsize=36)
    
    for j in range(output_dims):
        for i in range(input_dims):
            x_low = max(0.0, norm.ppf(0.01, loc=means[:,j,i], scale=scales[:,j,i]).min())
            x_high = norm.ppf(0.99, loc=means[:,j,i], scale=scales[:,j,i]).max()
            if isinstance(nyquist, np.ndarray):
                x_high = min(x_high, nyquist[j,i])

            x = np.linspace(x_low, x_high, 1000)
            psd = np.zeros(x.shape)

            for q in range(mixtures):
                single_psd = weights[q,j] * norm.pdf(x, loc=means[q,j,i], scale=scales[q,j,i])
                #single_psd = np.log(single_psd+1)
                axes[j,i].plot(x, single_psd, '--', c='r', zorder=2)
                psd += single_psd
           
            axes[j,i].plot(x, psd, 'k-', zorder=1)
            axes[j,i].set_yticks([])
            axes[j,i].set_ylim(0, None)
            if titles is not None:
                axes[j,i].set_title(titles[j])

    axes[output_dims-1,i].set_xlabel('Frequency')

    if filename is not None:
        plt.savefig(filename+'.pdf', dpi=300)
    if show:
        plt.show()
    return fig, axes

