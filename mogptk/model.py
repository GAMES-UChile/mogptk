import os
import sys
import time
import pickle
import numpy as np
import torch
import logging
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .serie import Serie
from .dataset import DataSet
from .gpr import GPR, CholeskyException, Kernel, MultiOutputKernel, IndependentMultiOutputKernel
from .errors import mean_absolute_error, mean_absolute_percentage_error, symmetric_mean_absolute_percentage_error, mean_squared_error, root_mean_squared_error

logger = logging.getLogger('mogptk')

def LoadModel(filename):
    """
    Load model from a given file that was previously saved with `model.save()`.

    Args:
        filename (str): File name to load from.

    Examples:
        >>> LoadModel('filename')
    """
    filename += ".npy" 
    with open(filename, 'rb') as r:
        return pickle.load(r)

class Exact:
    """
    Exact inference for Gaussian process regression.
    """
    def build(self, kernel, x, y, mean=None, name=None):
        return GPR(kernel, x, y, mean=mean, name=name)

class Model:
    def __init__(self, dataset, kernel, model=Exact(), mean=None, name=None, rescale_x=False):
        """
        Model is the base class for multi-output Gaussian process models.

        Args:
            dataset (mogptk.dataset.DataSet, mogptk.data.Data): `DataSet` with `Data` objects for all the channels. When a (list or dict of) `Data` object is passed, it will automatically be converted to a `DataSet`.
            kernel (mogptk.gpr.kernel.Kernel): The kernel class.
            model: Gaussian process model to use, such as `mogptk.model.Exact`.
            mean (mogptk.gpr.mean.Mean): The mean class.
            name (str): Name of the model.
            rescale_x (bool): Rescale the X axis to [0,1000] to help training.
        """
        
        if not isinstance(dataset, DataSet):
            dataset = DataSet(dataset)
        if dataset.get_output_dims() == 0:
            raise ValueError("dataset must have at least one channel")
        names = [name for name in dataset.get_names() if name is not None]
        if len(set(names)) != len(names):
            raise ValueError("all data channels must have unique names")

        if rescale_x:
            dataset.rescale_x()
        else:
            for channel in dataset:
                for dim in range(channel.get_input_dims()):
                    xran = np.max(channel.X[dim].transformed) - np.min(channel.X[dim].transformed)
                    if xran < 1e-3:
                        logger.warning("Very small X range may give problems, it is suggested to scale up your X axis")
                    elif 1e4 < xran:
                        logger.warning("Very large X range may give problems, it is suggested to scale down your X axis")

        self.name = name
        self.dataset = dataset
        self.kernel = kernel

        X = [[x[channel.mask] for x in channel.X] for channel in self.dataset]
        Y = [np.array(channel.Y[channel.mask]) for channel in self.dataset]
        x, y = self._to_kernel_format(X, Y)

        self.model = model.build(kernel, x, y, mean, name)
        if issubclass(type(kernel), MultiOutputKernel) and issubclass(type(model), Exact):
            self.model.noise.assign(0.0, lower=0.0, trainable=False)  # handled by MultiOutputKernel

    ################################################################

    def print_parameters(self):
        """
        Print the parameters of the model in a table.

        Examples:
            >>> model.print_parameters()
        """
        self.model.print_parameters()

    def get_parameters(self):
        """
        Returns all parameters of the kernel.

        Returns:
            list: mogptk.gpr.parameter.Parameter

        Examples:
            >>> params = model.get_parameters()
        """
        self.model.parameters()

    def save(self, filename):
        """
        Save the model to a given file that can then be loaded using `LoadModel()`.

        Args:
            filename (str): File name to save to, automatically appends '.npy'.

        Examples:
            >>> model.save('filename')
        """
        filename += ".npy" 
        try:
            os.remove(filename)
        except OSError:
            pass
        with open(filename, 'wb') as w:
            pickle.dump(self, w)

    def log_marginal_likelihood(self):
        """
        Returns the log marginal likelihood of the kernel and its data and parameters.

        Returns:
            float: The current log marginal likelihood.

        Examples:
            >>> model.log_marginal_likelihood()
        """
        return self.model.log_marginal_likelihood().detach().cpu().item()

    def loss(self):
        """
        Returns the loss of the kernel and its data and parameters.

        Returns:
            float: The current loss.

        Examples:
            >>> model.loss()
        """
        return self.model.loss().detach().cpu().item()

    def error(self, method='MAE'):
        """
        Returns the error of the kernel prediction with the removed data points in the data set.

        Args:
            method (str): Error calculation method, such as MAE, MAPE, sMAPE, MSE, or RMSE.

        Returns:
            float: The current error.

        Examples:
            >>> model.error()
        """
        X, Y_true = self.dataset.get_test_data()
        x, y_true  = self._to_kernel_format(X, Y_true)
        y_pred, _ = self.model.predict(x)
        if method.lower() == 'mae':
            return mean_absolute_error(y_true, y_pred)
        elif method.lower() == 'mape':
            return mean_absolute_percentage_error(y_true, y_pred)
        elif method.lower() == 'smape':
            return symmetric_mean_absolute_percentage_error(y_true, y_pred)
        elif method.lower() == 'mse':
            return mean_squared_error(y_true, y_pred)
        elif method.lower() == 'rmse':
            return root_mean_squared_error(y_true, y_pred)
        else:
            raise ValueError("valid error calculation methods are MAE, MAPE, and RMSE")

    def train(
        self,
        method='Adam',
        iters=500,
        verbose=False,
        error=None,
        plot=False,
        **kwargs):
        """
        Trains the model by optimizing the (hyper)parameters of the kernel to approach the training data.

        Args:
            method (str): Optimizer to use such as LBFGS, Adam, Adagrad, or SGD.
            iters (int): Number of iterations, or maximum in case of LBFGS optimizer.
            verbose (bool): Print verbose output about the state of the optimizer.
            error (str): Calculate prediction error for each iteration by the given method, such as MAE, MAPE, or RMSE.
            plot (bool): Plot the negative log likelihood.
            **kwargs (dict): Additional dictionary of parameters passed to the PyTorch optimizer. 

        Returns:
            numpy.ndarray: Losses for all iterations.
            numpy.ndarray: Errors for all iterations. Only if `error` is set, otherwise zero.

        Examples:
            >>> model.train()
            
            >>> model.train(method='lbfgs', tolerance_grad=1e-10, tolerance_change=1e-12)
            
            >>> model.train(method='adam', lr=0.5)
        """
        if error is not None and all(not channel.has_test_data() for channel in self.dataset):
            raise ValueError("data set must have test points (such as removed ranges) when error is specified")

        if method.lower() in ('l-bfgs', 'lbfgs', 'l-bfgs-b', 'lbfgsb'):
            method = 'LBFGS'
        elif method.lower() == 'adam':
            method = 'Adam'
        elif method.lower() == 'sgd':
            method = 'SGD'
        elif method.lower() == 'adagrad':
            method = 'AdaGrad'

        if verbose:
            training_points = sum([len(channel.get_train_data()[1]) for channel in self.dataset])
            parameters = sum([int(np.prod(param.shape)) for param in self.model.parameters()])
            print('\nStarting optimization using', method)
            print('‣ Model: {}'.format(self.name))
            print('‣ Channels: {}'.format(len(self.dataset)))
            if hasattr(self, 'Q'):
                print('‣ Mixtures: {}'.format(self.Q))
            print('‣ Training points: {}'.format(training_points))
            print('‣ Parameters: {}'.format(parameters))
            print('‣ Initial loss: {:.3g}'.format(self.loss()))
            if error is not None:
                print('‣ Initial error: {:.3g}'.format(self.error(error)))
            inital_time = time.time()

        losses = np.empty((iters+1,))
        errors = np.zeros((iters+1,))

        sys.__stdout__.write("\nStart %s:\n" % (method,))
        if method == 'LBFGS':
            if not 'max_iter' in kwargs:
                kwargs['max_iter'] = iters
                iters = 0
            optimizer = torch.optim.LBFGS(self.model.parameters(), **kwargs)

            def loss():
                i = int(optimizer.state_dict()['state'][0]['func_evals'])
                losses[i] = self.loss()
                if error is not None:
                    errors[i] = self.error(error)
                    if i % (kwargs['max_iter']/100) == 0:
                        sys.__stdout__.write("% 5d/%d  loss=%10g  error=%10g\n" % (i, kwargs['max_iter'], losses[i], errors[i]))
                elif i % (kwargs['max_iter']/100) == 0:
                    sys.__stdout__.write("% 5d/%d  loss=%10g\n" % (i, kwargs['max_iter'], losses[i]))
                return losses[i]
            optimizer.step(lambda: loss())
            iters = int(optimizer.state_dict()['state'][0]['func_evals'])
        else:
            if method == 'Adam':
                if 'lr' not in kwargs:
                    kwargs['lr'] = 0.1
                optimizer = torch.optim.Adam(self.model.parameters(), **kwargs)
            elif method == 'SGD':
                optimizer = torch.optim.SGD(self.model.parameters(), **kwargs)
            elif method == 'AdaGrad':
                optimizer = torch.optim.Adagrad(self.model.parameters(), **kwargs)
            else:
                print("Unknown optimizer:", method)

            for i in range(iters):
                losses[i] = self.loss()
                if error is not None:
                    errors[i] = self.error(error)
                    if i % (iters/100) == 0:
                        sys.__stdout__.write("% 5d/%d  loss=%10g  error=%10g\n" % (i, iters, losses[i], errors[i]))
                elif i % (iters/100) == 0:
                    sys.__stdout__.write("% 5d/%d  loss=%10g\n" % (i, iters, losses[i]))
                optimizer.step()
        losses[iters] = self.loss()
        if error is not None:
            errors[iters] = self.error(error)
            sys.__stdout__.write("% 5d/%d  loss=%10g  error=%10g\n" % (iters, iters, losses[iters], errors[iters]))
        else:
            sys.__stdout__.write("% 5d/%d  loss=%10g\n" % (iters, iters, losses[iters]))
        sys.__stdout__.write("Finished\n")

        if verbose:
            elapsed_time = time.time() - inital_time
            print('\nOptimization finished in {}'.format(_format_duration(elapsed_time)))
            print('‣ Function evaluations: {}'.format(iters))
            print('‣ Final loss: {:.3g}'.format(losses[iters]))
            if error is not None:
                print('‣ Final error: {:.3g}'.format(errors[iters]))

        self.iters = iters
        self.losses = losses
        self.errors = errors
        if plot:
            self.plot_losses()
        return losses, errors

    ################################################################################
    # Predictions ##################################################################
    ################################################################################

    # TODO: add get_prediction

    def _to_kernel_format(self, X, Y=None):
        """
        Return the data vectors in the format used by the kernels. If Y is not passed, than only X data is returned.

        Returns:
            numpy.ndarray: X data of shape (n,2) where X[:,0] contains the channel indices and X[:,1] the X values.
            numpy.ndarray: Y data.
            numpy.ndarray: Original but normalized X data. Only if no Y is passed.
        """
        if isinstance(X, dict):
            x_dict = X
            X = self.dataset.get_prediction()
            for name, channel_x in x_dict.items():
                X[self.dataset.get_index(name)] = channel_x
        elif isinstance(X, np.ndarray):
            X = list(X)
        elif not isinstance(X, list):
            raise ValueError("X must be a list, dict or numpy.ndarray")
        if len(X) != len(self.dataset.channels):
            raise ValueError("X must be a list of shape [(n,)] * input_dims for each channel")
        X_orig = X
        X = X.copy()
        for j, channel_x in enumerate(X):
            input_dims = self.dataset.get_input_dims()[j]
            if isinstance(channel_x, np.ndarray):
                if channel_x.ndim == 1:
                    channel_x = channel_x.reshape(-1, 1)
                if channel_x.ndim != 2 or channel_x.shape[1] != input_dims:
                    raise ValueError("X must be a list of shape (n,input_dims) or [(n,)] * input_dims for each channel")
                channel_x = [channel_x[:,i] for i in range(input_dims)]
            elif not isinstance(channel_x, list):
                raise ValueError("X must be a list of lists or numpy.ndarrays")
            if not all(isinstance(x, np.ndarray) for x in channel_x):
                raise ValueError("X must be a list of shape (n,input_dims) or [(n,)] * input_dims for each channel")
            X[j] = np.array([self.dataset[j].X[i].transform(channel_x[i]) for i in range(input_dims)]).T

        chan = [i * np.ones(len(X[i])) for i in range(len(X))]
        chan = np.concatenate(chan).reshape(-1, 1)
        if len(X) == 0:
            x = np.array([])
        else:
            x = np.concatenate(X, axis=0)
            x = np.concatenate([chan, x], axis=1)
        if Y is None:
            return x, X_orig

        if isinstance(Y, np.ndarray):
            Y = list(Y)
        elif not isinstance(Y, list):
            raise ValueError("Y must be a list or numpy.ndarray")
        if len(Y) != len(self.dataset.channels):
            raise ValueError("Y must be a list of shape (n,) for each channel")
        Y = Y.copy()
        for j, channel_y in enumerate(Y):
            if channel_y.ndim != 1:
                raise ValueError("Y must be a list of shape (n,) for each channel")
            if channel_y.shape[0] != X[j].shape[0]:
                raise ValueError("Y must have the same number of data points per channel as X")
            Y[j] = self.dataset[j].Y.transform(channel_y, x=X_orig[j])
        if len(Y) == 0:
            y = np.array([])
        else:
            y = np.concatenate(Y, axis=0).reshape(-1, 1)
        return x, y

    def predict(self, X=None, sigma=2.0, transformed=False):
        """
        Predict using the prediction range of the data set and save the prediction in that data set. Otherwise, if `X` is passed, use that as the prediction range and return the prediction instead of saving it.

        Args:
            X (list, dict): Dictionary where keys are channel index and elements numpy arrays with channel inputs. If passed, results will be returned and not saved in the data set for later retrieval.
            sigma (float): The confidence interval's number of standard deviations.
            transformed (boolean): Return transformed data as used for training.

        Returns:
            numpy.ndarray: Y mean prediction of shape (n,).
            numpy.ndarray: Y lower prediction of uncertainty interval of shape (n,).
            numpy.ndarray: Y upper prediction of uncertainty interval of shape (n,).

        Examples:
            >>> model.predict(plot=True)
        """
        save = X is None
        if save and transformed:
            raise ValueError('must pass an X range explicitly in order to return transformed data')
        if save:
            X = self.dataset.get_prediction_x()
        x, X = self._to_kernel_format(X)

        mu, var = self.model.predict(x)

        i = 0
        Mu = []
        Var = []
        Lower = []
        Upper = []
        for j in range(self.dataset.get_output_dims()):
            N = X[j][0].shape[0]
            Mu.append(np.squeeze(mu[i:i+N]))
            Var.append(np.squeeze(var[i:i+N]))
            Lower.append(Mu[j] - sigma*np.sqrt(Var[j]))
            Upper.append(Mu[j] + sigma*np.sqrt(Var[j]))
            i += N

        if save:
            for j in range(self.dataset.get_output_dims()):
                self.dataset[j].Y_mu_pred[self.name] = Mu[j]
                self.dataset[j].Y_var_pred[self.name] = Var[j]

        if transformed:
            return Mu, Lower, Upper
        else:
            for j in range(self.dataset.get_output_dims()):
                Mu[j] = self.dataset[j].Y.detransform(Mu[j], X[j])
                Lower[j] = self.dataset[j].Y.detransform(Lower[j], X[j])
                Upper[j] = self.dataset[j].Y.detransform(Upper[j], X[j])
            return Mu, Lower, Upper

    def plot_losses(self, title=None, figsize=None, legend=True, errors=True):
        if not hasattr(self, 'losses'):
            raise Exception("must be trained in order to plot the losses")

        if figsize is None:
            figsize = (12,3)

        fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
        ax.plot(np.arange(0,self.iters+1), self.losses[:self.iters+1], c='k', ls='-')
        ax.set_xlim(0, self.iters)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')

        legends = []
        legends.append(plt.Line2D([0], [0], ls='-', color='k', label='Loss'))
        if errors and hasattr(self, 'errors'):
            ax2 = ax.twinx()
            ax2.plot(np.arange(0,self.iters+1), self.errors[:self.iters+1], c='k', ls='-.')
            ax2.set_ylabel('Error')
            legends.append(plt.Line2D([0], [0], ls='-.', color='k', label='Error'))

        if title is not None:
            fig.suptitle(title, fontsize=18)

        if legend:
            ax.legend(handles=legends)

    def plot_prediction(self, title=None, figsize=None, legend=True, transformed=False):
        """
        Plot the data including removed observations, latent function, and predictions of this model for each channel.

        Args:
            title (str): Set the title of the plot.
            figsize (tuple): Set the figure size.
            legend (boolean): Disable legend.
            transformed (boolean): Display transformed Y data as used for training.

        Returns:
            matplotlib.figure.Figure: The figure.
            list of matplotlib.axes.Axes: List of axes.

        Examples:
            >>> fig, axes = dataset.plot(title='Title')
        """
        return self.dataset.plot(pred=self.name, title=title, figsize=figsize, legend=legend, transformed=transformed)

    def get_gram_matrix(self, start=None, end=None, n=31):
        """
        Returns the gram matrix evaluated between `start` and `end` with `n` number of points. If `start` and `end` are not set, the minimum and maximum X points of the data are used.

        Args:
            start (float, list, array): Interval minimum.
            end (float, list, array): Interval maximum.
            n (int): Number of points per channel.

        Returns:
            numpy.ndarray: Array of shape (n,n).

        Examples:
            >>> model.get_gram_matrix()
        """
        if start is None:
            start = [np.array(data.X[0].transformed).min() for data in self.dataset]
        if end is None:
            end = [np.array(data.X[0].transformed).max() for data in self.dataset]

        M = len(self.dataset)
        if not isinstance(start, (list, np.ndarray)):
            start = [start] * M
        if not isinstance(end, (list, np.ndarray)):
            end = [end] * M

        X = np.zeros((M*n, 2))
        X[:,0] = np.repeat(np.arange(M), n)
        for m in range(M):
            if n== 1:
                X[m*n:(m+1)*n,1] = np.array((start[m]+end[m])/2.0)
            else:
                X[m*n:(m+1)*n,1] = np.linspace(start[m], end[m], n)

        return self.model.K(X)

    def plot(self, start=None, end=None, n=31, title=None, figsize=(12,12)):
        """
        Plot the gram matrix of associated kernel.

        Args:
            start (float, list, array): Interval minimum.
            end (float, list, array): Interval maximum.
            n (int): Number of points per channel.
            title (str): Figure title.
            figsize (tuple): Figure size.

        Returns:
            figure: Matplotlib figure.
            axis: Matplotlib axis.
        """
        K_gram = self.get_gram_matrix(start, end, n)
            
        fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
        if title is not None:
            fig.suptitle(title, fontsize=18)

        color_range = np.abs(K_gram).max()
        norm = matplotlib.colors.Normalize(vmin=-color_range, vmax=color_range)
        im = ax.matshow(K_gram, cmap='coolwarm', norm=norm)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.3)
        fig.colorbar(im, cax=cax)

        # Major ticks every 20, minor ticks every 5
        M = len(self.dataset)
        major_ticks = np.arange(-0.5, M * n, n)
        minor_ticks = np.arange(-0.5, M * n, 2)

        ax.set_xticks(major_ticks)
        ax.set_yticks(major_ticks)
        ax.grid(which='major', lw=1.5, c='k')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='both', length=0)
        return fig, ax

def _format_duration(s):
    s = round(s)
    days = int(s/86400)
    hours = int(s%86400/3600)
    minutes = int(s%3600/60)
    seconds = int(s%60)

    duration = ''
    if 1 < days:
        duration += ' %d days' % days
    elif days == 1:
        duration += ' 1 day'
    if 1 < hours:
        duration += ' %d hours' % hours
    elif hours == 1:
        duration += ' 1 hour'
    if 1 < minutes:
        duration += ' %d minutes' % minutes
    elif minutes == 1:
        duration += ' 1 minute'
    if 1 < seconds:
        duration += ' %d seconds' % seconds
    elif days == 1:
        duration += ' 1 second'
    else:
        duration += ' less than one second'
    return duration[1:]
