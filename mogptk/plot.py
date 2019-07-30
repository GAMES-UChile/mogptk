import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

def plot(model, filename=None, title=None):
    """plot will plot the model in graphs per input and output dimensions. Output dimensions will stack the graphs vertically while input dimensions stacks them horizontally. Optionally, you can output the figure to a file and set a title."""
    data = model.data
    channels = range(data.get_output_dims())

    sns.set(font_scale=2)
    sns.axes_style("darkgrid")
    sns.set_style("whitegrid")

    fig, axes = plt.subplots(len(channels), data.get_input_dims(), figsize=(20, len(channels)*5), sharey=False, constrained_layout=True, squeeze=False)
    if title != None:
        fig.suptitle(title, fontsize=36)

    plotting_pred = False
    plotting_F = False
    plotting_all_obs = False
    for channel in channels:
        if channel in model.Y_mu_pred:
            lower = model.Y_mu_pred[channel] - model.Y_var_pred[channel]
            upper = model.Y_mu_pred[channel] + model.Y_var_pred[channel]

            for i in range(data.get_input_dims()):
                axes[channel, i].plot(model.X_pred[channel][:,i], model.Y_mu_pred[channel], 'b-', lw=3)
                axes[channel, i].fill_between(model.X_pred[channel][:,i], lower, upper, color='b', alpha=0.1)
                axes[channel, i].plot(model.X_pred[channel][:,i], lower, 'b-', lw=1, alpha=0.5)
                axes[channel, i].plot(model.X_pred[channel][:,i], upper, 'b-', lw=1, alpha=0.5)
            plotting_pred = True

        if channel in data.F:
            for i in range(data.get_input_dims()):
                n = (len(data.X[channel][:,i]) + len(model.X_pred[channel][:,i]))*10
                x_min = np.min(np.concatenate((data.X[channel][:,i], model.X_pred[channel][:,i])))
                x_max = np.max(np.concatenate((data.X[channel][:,i], model.X_pred[channel][:,i])))

                x = np.zeros((n, data.get_input_dims())) # assuming other input dimensions are zeros
                x[:,i] = np.linspace(x_min, x_max, n)
                y = data.F[channel](x)

                axes[channel, i].plot(x[:,i], y, 'r--', lw=1)
                plotting_F = True

        X_removed, Y_removed = data.get_del_obs(channel)
        if len(X_removed) > 0:
            for i in range(data.get_input_dims()):
                axes[channel, i].plot(X_removed[:,i], Y_removed, 'rx', mew=2, ms=10)
            plotting_all_obs = True

        for i in range(data.get_input_dims()):
            axes[channel, i].plot(data.X[channel][:,i], data.Y[channel], 'kx', mew=2, ms=10)
        
    for i in range(data.get_input_dims()):
        axes[0, i].set_title('Input dimension %d' % (i))

    # build legend
    legend = []
    legend.append(plt.Line2D([0], [0], ls='', marker='x', color='k', mew=2, ms=10, label='Training observations'))
    if plotting_all_obs:
        legend.append(plt.Line2D([0], [0], ls='', marker='x', color='r', mew=2, ms=10, label='Removed observations'))
    if plotting_F:
        legend.append(plt.Line2D([0], [0], ls='--', color='r', label='Latent function'))
    if plotting_pred:
        legend.append(plt.Line2D([0], [0], ls='-', color='b', lw=3, label='Prediction'))
    plt.legend(handles=legend, loc='best')

    if filename != None:
        plt.savefig(filename+'.pdf', dpi=300)
    plt.show()

def plot_sm_psd(model, title='', filename=None):
    sns.set(font_scale=2)
    sns.axes_style("darkgrid")
    sns.set_style("whitegrid")

    fig, axes = plt.subplots(1, 1, figsize=(20, 5), sharey=False, constrained_layout=True, squeeze=False)
    if title != None:
        fig.suptitle(title, fontsize=36)

    nyquist = model.data.get_nyquist_estimation()[0,0]
    means = model._get_param_across('mixture_means').reshape(-1)
    weights = model._get_param_across('mixture_weights').reshape(-1)
    scales = model._get_param_across('mixture_scales').reshape(-1)

    means *= np.pi * 2
    
    # calculate bounds
    x_low = max(0.0, norm.ppf(0.001, loc=means, scale=scales).min())
    x_high = min(nyquist, norm.ppf(0.99, loc=means, scale=scales).max())
    x = np.linspace(x_low, x_high, 1000)
    psd = np.zeros(x.shape)
    
    for q in range(model.Q):
        single_psd = weights[q] * norm.pdf(x, loc=means[q], scale=scales[q])
        axes[0,0].plot(x, single_psd, '--', c='k', zorder=2)
        psd += single_psd
    
    axes[0,0].plot(x, psd, 'b-', lw=3, zorder=1)
    axes[0,0].set_xlabel(r'$\omega$')
    axes[0,0].set_ylabel('PSD')

    if filename != None:
        plt.savefig(filename+'.pdf', dpi=300)
    plt.show()

def plot_psd(model, title='', filename=None):
    """
    Plot power spectral density of 
    single output GP-SM
    """
    means = model._get_param_across('mixture_means').reshape(-1)
    weights = model._get_param_across('mixture_weights').reshape(-1)
    scales = model._get_param_across('mixture_scales').reshape(-1)
    
    # calculate bounds
    x_low = norm.ppf(0.001, loc=means, scale=scales).min()
    x_high = norm.ppf(0.99, loc=means, scale=scales).max()
    
    x = np.linspace(x_low, x_high + 1, 1000)
    psd = np.zeros_like(x)
    
    for q in range(model.Q):
        single_psd = weights[q] * norm.pdf(x, loc=means[q], scale=scales[q])
        plt.plot(x, single_psd, '--', lw=1.2, c='orange', zorder=2, alpha=0.9)
        plt.axvline(means[q], ymin=0.001, ymax=0.1, lw=2, color='grey')
        psd = psd + single_psd
        
    # symmetrize PSD
    if psd[x<0].size != 0:
        psd = psd + np.r_[psd[x<0][::-1], np.zeros((x>=0).sum())]
        
    plt.plot(x, psd, lw=2.5, c='r', alpha=0.7, zorder=1)
    plt.xlim(0, x[-1] + 0.1)
    # plt.yscale('log')
    plt.xlabel(r'$\omega$')
    plt.ylabel('PSD')
    plt.title(title)
    plt.show()

    if filename != None:
        plt.savefig(filename+'.pdf', dpi=300)



