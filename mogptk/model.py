import os
import time
import pickle
import numpy as np
import torch
from .serie import Serie
from .dataset import DataSet
from .kernels import GPR
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

import logging
logger = logging.getLogger('mogptk')

eps = 1e-20

def LoadModel(filename):
    """
    Load model from a given file that was previously saved with `model.save()`.

    Args:
        filename (str): Filename to load from.

    Examples:
        >>> Load('filename')
    """
    filename += ".npy" 
    with open(filename, 'rb') as r:
        return pickle.load(r)

class Exact:
    def build(self, kernel, x, y, name=None):
        return GPR(kernel, x, y, name=name)

class Model:
    def __init__(self, dataset, kernel, model=Exact(), name=None):
        """
        Base class for Multi-Output Gaussian process models. See subclasses for instantiation.

        Args:
            dataset (mogptk.dataset.DataSet, mogptk.data.Data): DataSet with Data objects for all the channels. When a (list or dict of) Data object is passed, it will automatically be converted to a DataSet.
            name (str): Name of the model.
        """
        
        if not isinstance(dataset, DataSet):
            dataset = DataSet(dataset)
        if dataset.get_output_dims() == 0:
            raise Exception("dataset must have at least one channel")
        names = [name for name in dataset.get_names() if name is not None]
        if len(set(names)) != len(names):
            raise Exception("all data channels must have unique names")

        for channel in dataset:
            for dim in range(channel.get_input_dims()):
                xran = np.max(channel.X[dim].transformed) - np.min(channel.X[dim].transformed)
                if xran < 1e-3:
                    logger.warning("Very small X range may give problems, it is suggested to scale up your X-axis")
                elif 1e4 < xran:
                    logger.warning("Very large X range may give problems, it is suggested to scale down your X-axis")

        self.name = name
        self.dataset = dataset
        self.kernel = kernel

        X = [np.array([x[channel.mask] for x in channel.X]).T for channel in self.dataset.channels]
        Y = [np.array(channel.Y[channel.mask]) for channel in self.dataset.channels]
        x, y = self._to_kernel_format(X, Y)

        self.model = model.build(kernel, x, y, name)

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
        Returns all parameters set for the kernel per component.

        Examples:
            >>> params = model.get_parameters()
        """
        self.model.parameters()

    def save(self, filename):
        """
        Save model to a given file that can then be loaded with `Load()`.

        Args:
            filename (str): Filename to save to, automatically appends '.model'.

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
        return self.model.log_marginal_likelihood().detach().item()

    def train(
        self,
        method='Adam',
        iters=500,
        verbose=False,
        **kwargs):
        """
        Trains the model using the kernel and its parameters.

        For different optimizers, see scipy.optimize.minimize.
        It can be bounded by a maximum number of iterations, disp will output final
        optimization information. When using the 'Adam' optimizer, a
        learning_rate can be set.

        Args:
            method (str): Optimizer to use such as LBFGS, Adam, Adagrad, or SGD. Defaults to Adam.
            iters (int): Number of iterations, or maximum in case of LBFGS optimizer. Defaults to 500.
            verbose (bool): Print verbose output about the state of the optimizer.
            **kwargs (dict): Additional dictionary of parameters passed to the PyTorch optimizer. 

        Examples:
            >>> model.train()
            
            >>> model.train(method='lbfgs', tolerance_grad=1e-10, tolerance_change=1e-12)
            
            >>> model.train(method='adam', lr=1e-4)
        """
        if verbose:
            training_points = sum([len(channel.get_train_data()[0]) for channel in self.dataset])
            parameters = sum([int(np.prod(param.shape)) for param in self.model.parameters()])
            print('\nStarting optimization using', method)
            print('‣ Model: {}'.format(self.name))
            print('‣ Channels: {}'.format(len(self.dataset)))
            if hasattr(self, 'Q'):
                print('‣ Mixtures: {}'.format(self.Q))
            print('‣ Training points: {}'.format(training_points))
            print('‣ Parameters: {}'.format(parameters))
            print('‣ Initial NLL: {:.3f}'.format(-self.model.log_marginal_likelihood().tolist()))
            inital_time = time.time()

        try:
            if method.lower() in ('l-bfgs', 'lbfgs', 'l-bfgs-b', 'lbfgsb'):
                if not 'max_iter' in kwargs:
                    kwargs['max_iter'] = iters
                    iters = 0
                optimizer = torch.optim.LBFGS(self.model.parameters(), **kwargs)
                optimizer.step(lambda: self.model.loss())
                iters = optimizer.state_dict()['state'][0]['func_evals']
            elif method.lower() == 'adam':
                if not 'lr' in kwargs:
                    kwargs['lr'] = 0.1
                optimizer = torch.optim.Adam(self.model.parameters(), **kwargs)
                for i in range(iters):
                    loss = self.model.loss()
                    optimizer.step()
            elif method.lower() == 'sgd':
                optimizer = torch.optim.SGD(self.model.parameters(), **kwargs)
                for i in range(iters):
                    loss = self.model.loss()
                    optimizer.step()
            elif method.lower() == 'adagrad':
                optimizer = torch.optim.Adagrad(self.model.parameters(), **kwargs)
                for i in range(iters):
                    loss = self.model.loss()
                    optimizer.step()
            else:
                print("Unknown optimizer:", method)
        except Exception as e:
            return

        if verbose:
            elapsed_time = time.time() - inital_time
            print('\nOptimization finished in {:.2f} minutes'.format(elapsed_time / 60.0))
            print('‣ Function evaluations: {}'.format(iters))
            print('‣ Final NLL: {:.3f}'.format(-self.model.log_marginal_likelihood().tolist()))

    ################################################################################
    # Predictions ##################################################################
    ################################################################################

    # TODO: add get_prediction

    def _to_kernel_format(self, X, Y=None):
        """
        Return the data vectors in the format as used by the kernels.

        Returns:
            numpy.ndarray: X data of shape (n,2) where X[:,0] contains the channel indices and X[:,1] the X values.
            numpy.ndarray: Y data.
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
            raise ValueError("X must be a list of shape (n,input_dims) for each channel")
        X_orig = [channel_x.copy() for channel_x in X]
        for j, channel_x in enumerate(X):
            input_dims = self.dataset.get_input_dims()[j]
            if isinstance(channel_x, list):
                channel_x = np.array(channel_x)
            elif not isinstance(channel_x, np.ndarray):
                raise ValueError("X must be a list of lists or numpy.ndarrays")
            if channel_x.ndim == 1:
                channel_x = channel_x.reshape(-1, 1)
            if channel_x.ndim != 2 or channel_x.shape[1] != input_dims:
                raise ValueError("X must be a list of shape (n,input_dims) for each channel")
            X[j] = np.array([self.dataset[j].X[i].transform(channel_x[:,i]) for i in range(input_dims)]).T

        chan = [i * np.ones(len(X[i])) for i in range(len(X))]
        chan = np.concatenate(chan).reshape(-1, 1)
        if len(X) == 0:
            x = np.array([])
        else:
            x = np.concatenate(X, axis=0)
            x = np.concatenate([chan, x], axis=1)
        if Y is None:
            return x

        if isinstance(Y, np.ndarray):
            Y = list(Y)
        elif not isinstance(Y, list):
            raise ValueError("Y must be a list or numpy.ndarray")
        if len(Y) != len(self.dataset.channels):
            raise ValueError("Y must be a list of shape (n,) for each channel")
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
        Predict with model.

        Will make a prediction using x as input. If no input value is passed, the prediction will 
        be made with atribute self.X_pred that can be setted with other functions.

        Args:
            X (list, dict, optional): Dictionary where keys are channel index and elements numpy arrays with channel inputs. If passed, results will be returned and not saved in the data set for later retrieval.
            sigma (float, optional): The uncertainty interval's number of standard deviations.
            transformed (boolean, optional): Return transformed data as used for training.

        Returns:
            numpy.ndarray: Y mean prediction of shape (n,).
            numpy.ndarray: Y lower prediction of uncertainty interval of shape (n,).
            numpy.ndarray: Y upper prediction of uncertainty interval of shape (n,).

        Examples:
            >>> model.predict(plot=True)
        """
        save = X is None
        if save:
            X = self.dataset.get_prediction_x()
        x = self._to_kernel_format(X)

        mu, var = self.model.predict(x)

        i = 0
        Mu = []
        Var = []
        for j in range(self.dataset.get_output_dims()):
            N = X[j].shape[0]
            Mu.append(np.squeeze(mu[i:i+N]))
            Var.append(np.squeeze(var[i:i+N]))
            i += N

        if save:
            for j in range(self.dataset.get_output_dims()):
                #self.dataset[j].X_pred = [Serie(X[j][:,i], self.dataset[j].X[i].transformers) for i in range(self.dataset[j].get_input_dims())]
                self.dataset[j].Y_mu_pred[self.name] = Mu[j]
                self.dataset[j].Y_var_pred[self.name] = Var[j]
        else:
            Lower = []
            Upper = []
            for j in range(self.dataset.get_output_dims()):
                Lower.append(Mu[j] - sigma*np.sqrt(Var[j]))
                Upper.append(Mu[j] + sigma*np.sqrt(Var[j]))

            X_pred = [np.array([self.dataset[j].X[i].transform(X[j][:,i]) for i in range(self.dataset[j].get_input_dims())]) for channel in range(self.dataset.get_output_dims())]
            if transformed:
                return X_pred, Mu, Lower, Upper
            else:
                for j in range(self.dataset.get_output_dims()):
                    Mu[j] = self.dataset[j].Y.detransform(Mu[j], X[j])
                    Lower[j] = self.dataset[j].Y.detransform(Lower[j], X[j])
                    Upper[j] = self.dataset[j].Y.detransform(Upper[j], X[j])
                return X, Mu, Lower, Upper

    def plot(self, xmin=None, xmax=None, n_points=31, title=None, figsize=(12,12)):
        """
        Plot the gram matrix of associated kernel.

        The gram matrix is evaluated depending a equally spaced grid 
        between [xmin_i, xmax_i] for i = 0, ..., n_channels.

        Args:
            xmin (float, list, array): Interval minimum.
            xmax (float, list, array): Interval maximum.
            n_points (int): Number of points per channel.
            title (str): Figure title.
            figsize (tuple): Figure size.
        
        Returns:
            fig: Matplotlib figure.
            ax: Matplotlib axis.
        """
        if xmin is None:
            xmin = [np.array(data.X[0].transformed).min() for data in self.dataset]
        if xmax is None:
            xmax = [np.array(data.X[0].transformed).max() for data in self.dataset]

        M = len(self.dataset)
        if not isinstance(xmin, (list, np.ndarray)):
            xmin = [xmin] * M
        if not isinstance(xmax, (list, np.ndarray)):
            xmax = [xmax] * M

        X = np.zeros((M*n_points, 2))
        X[:,0] = np.repeat(np.arange(M), n_points)
        for m in range(M):
            X[m*n_points:(m+1)*n_points,1] = np.linspace(xmin[m], xmax[m], n_points)
            
        fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
        if title is not None:
            fig.suptitle(title, fontsize=18)

        K_gram = self.model.K(X)
        color_range = np.abs(K_gram).max()
        norm = mpl.colors.Normalize(vmin=-color_range, vmax=color_range)
        im = ax.matshow(K_gram, cmap='coolwarm', norm=norm)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.3)
        fig.colorbar(im, cax=cax)

        # Major ticks every 20, minor ticks every 5
        major_ticks = np.arange(-0.5, M * n_points, n_points)
        minor_ticks = np.arange(-0.5, M * n_points, 2)

        ax.set_xticks(major_ticks)
        ax.set_yticks(major_ticks)
        ax.grid(which='major', lw=1.5, c='k')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='both', length=0)

        return fig, ax

