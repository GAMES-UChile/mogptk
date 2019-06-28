import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot(model, filename=None, title=None):
    data = model.data

    channels = range(data.get_output_dimensions())
    for channel in channels:
        input_dims = data.get_input_dimensions()
        if input_dims == 0:
            channels.remove(channel)
        elif input_dims != 1:
            raise Exception('only one dimensions input data can be plotted')

    sns.set(font_scale=2)
    sns.axes_style("darkgrid")
    sns.set_style("whitegrid")

    fig, axes = plt.subplots(len(channels), 1, figsize=(20, len(channels)*5), sharey=False, constrained_layout=True, squeeze=False)
    if title != None:
        fig.suptitle(title, fontsize=36)

    plotting_pred = False
    plotting_F = False
    plotting_all_obs = False
    for channel in channels:
        x_min = data.X_all[channel][0]
        x_max = data.X_all[channel][-1]
        
        if channel in model.Y_mu_pred:
            x_min = min(x_min, model.X_pred[channel][0])
            x_max = max(x_max, model.X_pred[channel][-1])

            lower = model.Y_mu_pred[channel] - model.Y_var_pred[channel]
            upper = model.Y_mu_pred[channel] + model.Y_var_pred[channel]

            axes[channel, 0].plot(model.X_pred[channel], model.Y_mu_pred[channel], 'b-', lw=3)
            axes[channel, 0].fill_between(model.X_pred[channel], lower, upper, color='b', alpha=0.1)
            axes[channel, 0].plot(model.X_pred[channel], lower, 'b-', lw=1, alpha=0.5)
            axes[channel, 0].plot(model.X_pred[channel], upper, 'b-', lw=1, alpha=0.5)
            plotting_pred = True

        if channel in data.F:
            x = np.arange(x_min, x_max+0.01, 0.01)
            y = data.F[channel](x)
            axes[channel, 0].plot(x, y, 'r--', lw=1)
            plotting_F = True

        js = []
        for i in range(len(data.X[channel])):
            x = data.X[channel][i]
            y = data.Y[channel][i]
            j = np.where(data.X_all[channel] == x)[0]
            if len(j) == 1 and data.Y_all[channel][j[0]] == y:
                js.append(j[0])

        X_removed = np.delete(data.X_all[channel], js)
        Y_removed = np.delete(data.Y_all[channel], js)
        if len(X_removed) > 0:
            axes[channel, 0].plot(X_removed, Y_removed, 'rx', mew=2, ms=10)
            plotting_all_obs = True

        axes[channel, 0].plot(data.X[channel], data.Y[channel], 'kx', mew=2, ms=10)

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
        plt.savefig(filename, dpi=300)
    plt.show()

