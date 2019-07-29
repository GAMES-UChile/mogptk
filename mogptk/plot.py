import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
            x = np.arange(x_min, x_max+0.01, 0.01)
            y = data.F[channel](x) # TODO: multi input dims
            axes[channel, 0].plot(x, y, 'r--', lw=1)
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
        plt.savefig(filename, dpi=300)
    plt.show()

