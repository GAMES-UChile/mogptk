import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

class Prediction:
    def __init__(self, model):
        self.model = model
        self.data = model.data
        self.X = {}
        self.Y_mu = {}
        self.Y_var = {}

    def get(self):
        """
        Returns the input, posterior mean and posterior variance values all channels.

        Returns:
            x_pred, y_mu_pred, y_var_pred: ndarrays with the input, posterior mean and 
                posterior variance of the las prediction done. 
        """
        if len(self.X) == 0:
            raise Exception("use predict before retrieving the predictions on the model")
        return self.X, self.Y_mu, self.Y_var

    def get_channel(self, channel):
        """
        Returns the input, posterior mean and posterior variance values for a given channel.

        Args:
            channel (str, int): Channel to set prediction, can be either a string with the name
                of the channel or a integer with the index.

        Returns:
            x_pred, y_mu_pred, y_var_pred: ndarrays with the input, posterior mean and 
                posterior variance of the las prediction done. 
        """
        if len(self.X) == 0:
            raise Exception("use predict before retrieving the predictions on the model")
        channel = self.data.get_channel_index(channel)

        return self.X[channel], self.Y_mu[channel], self.Y_var[channel]

    def set_range(self, channel, start=None, end=None, step=None, n=None):
        """
        Sets the prediction range for a certain channel in the interval [start,end].
        with either a stepsize step or a number of points n.

        Args:
            channel (str, int): Channel to set prediction, can be either a string with the name
                of the channel or a integer with the index.

            start (float, optional): Initial value of range, if not passed the first point of training
                data is taken. Default to None.

            end (float, optional): Final value of range, if not passed the last point of training
                data is taken. Default to None.

            step (float, optional): Spacing between values.

            n (int, optional): Number of samples to generate.

            If neither "step" or "n" is passed, default number of points is 100.
        """
        channel = self.data.get_channel_index(channel)

        if start == None:
            start = self.data.X[channel][0]
        elif isinstance(start, list):
            for i in range(self.data.get_input_dims()):
                start[i] = self.data.formatters[channel][i].parse(start[i])
        else:
            start = self.data.formatters[channel][0].parse(start)

        if end == None:
            end = self.data.X[channel][-1]
        elif isinstance(end, list):
            for i in range(self.data.get_input_dims()):
                end[i] = self.data.formatters[channel][i].parse(end[i])
        else:
            end = self.data.formatters[channel][0].parse(end)
        
        start = self.data._normalize_input_dims(start)
        end = self.data._normalize_input_dims(end)

        # TODO: works for multi input dims?
        if end <= start:
            raise Exception("start must be lower than end")

        # TODO: prediction range for multi input dimension; fix other axes to zero so we can plot?
        if step == None and n != None:
            self.X[channel] = np.empty((n, self.data.get_input_dims()))
            for i in range(self.data.get_input_dims()):
                self.X[channel][:,i] = np.linspace(start[i], end[i], n)
        else:
            if self.data.get_input_dims() != 1:
                raise Exception("cannot use step for multi dimensional input, use n")
            if step == None:
                step = (end[0]-start[0])/100
            else:
                step = self.data.formatters[channel][0].parse(step)
            self.X[channel] = np.arange(start[0], end[0]+step, step).reshape(-1, 1)
    
    def set(self, xs):
        """
        Sets the prediction ranges for all channels.

                Args:

                xs (list, dict): Prediction ranges, where the index/key is the channel
                        ID/name and the values are either lists or Numpy arrays.
        """
        if isinstance(xs, list):
            for channel in range(self.data.get_output_dims()):
                self.set(channel, xs[channel,:])
        elif isinstance(xs, dict):
            for channel, x in xs.items():
                self.set(channel, x)
        else:
            raise Exception("xs expected to be a list, dict or Numpy array")

    def set_channel(self, channel, x):
        """
        Sets the prediction range using a list of Numpy array for a certain channel.

        Args:
            channel (str, int): Channel to set prediction, can be either a string with the name
                of the channel or a integer with the index.
            x (ndarray): Numpy array with input values for channel.
        """
        if x.ndim != 2 or x.shape[1] != self.data.get_input_dims():
            raise Exception("x shape must be (n,input_dims)")

        channel = self.data.get_channel_index(channel)
        if isinstance(x, list):
            x = np.array(x)
        elif not isinstance(x, np.ndarray):
            raise Exception("x expected to be a list or Numpy array")

        self.X[channel] = x

    def predict(self):
        self.Y_mu, self.Y_var = self.model.predict(self.X)

    # TODO: keep in or out?
    def plot(self, filename=None, title=None):
        """
        Plot the model in graphs per input and output dimensions.

        Output dimensions will stack the graphs vertically while input dimensions stacks them horizontally.
        Optionally, you can output the figure to a file and set a title.
        """
        data = self.data
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
            if channel in self.Y_mu:
                lower = self.Y_mu[channel] - self.Y_var[channel]
                upper = self.Y_mu[channel] + self.Y_var[channel]

                for i in range(data.get_input_dims()):
                    axes[channel, i].plot(self.X[channel][:,i], self.Y_mu[channel], 'b-', lw=3)
                    axes[channel, i].fill_between(self.X[channel][:,i], lower, upper, color='b', alpha=0.1)
                    axes[channel, i].plot(self.X[channel][:,i], lower, 'b-', lw=1, alpha=0.5)
                    axes[channel, i].plot(self.X[channel][:,i], upper, 'b-', lw=1, alpha=0.5)
            
                    axes[channel, i].set_xlabel(data.input_labels[channel][i])
                    axes[channel, i].set_ylabel(data.output_labels[channel])
                    axes[channel, i].set_title(data.channel_names[channel], fontsize=30)

                    formatter = matplotlib.ticker.FuncFormatter(lambda x,pos: data.formatters[channel][i].format(x))
                    axes[channel, i].xaxis.set_major_formatter(formatter)
                plotting_pred = True

                if channel in data.F:
                    for i in range(data.get_input_dims()):
                        n = (len(data.X[channel][:,i]) + len(self.X[channel][:,i]))*10
                        x_min = np.min(np.concatenate((data.X[channel][:,i], self.X[channel][:,i])))
                        x_max = np.max(np.concatenate((data.X[channel][:,i], self.X[channel][:,i])))

                        x = np.zeros((n, data.get_input_dims())) # assuming other input dimensions are zeros
                        x[:,i] = np.linspace(x_min, x_max, n)
                        y = data.F[channel](x)

                        axes[channel, i].plot(x[:,i], y, 'r--', lw=1)
                        plotting_F = True

            X_removed, Y_removed = data.get_del_obs(channel)
            if len(X_removed) > 0:
                for i in range(data.get_input_dims()):
                    axes[channel, i].plot(X_removed[:,i], Y_removed, 'r.', mew=2, ms=8)
                plotting_all_obs = True

            for i in range(data.get_input_dims()):
                axes[channel, i].plot(data.X[channel][:,i], data.Y[channel], 'k.', mew=2, ms=8)
            
        # build legend
        legend = []
        legend.append(plt.Line2D([0], [0], ls='', marker='.', color='k', mew=2, ms=8, label='Training'))
        if plotting_all_obs:
            legend.append(plt.Line2D([0], [0], ls='', marker='.', color='r', mew=2, ms=8, label='Removed'))
        if plotting_F:
            legend.append(plt.Line2D([0], [0], ls='--', color='r', label='Latent'))
        if plotting_pred:
            legend.append(plt.Line2D([0], [0], ls='-', color='b', lw=3, label='Prediction'))
        plt.legend(handles=legend, loc='best')

        if filename != None:
            plt.savefig(filename+'.pdf', dpi=300)
        plt.show()
