import os
import time
import pickle
import numpy as np
import torch
from .dataset import DataSet
from .kernels import GPR
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tabulate import tabulate

import logging
logger = logging.getLogger('mogptk')

eps = 1e-20

def Load(filename):
    """
    Load model from a given file that was previously saved with `model.save()`.

    Args:
        filename (str): Filename to load from.

    Examples:
        >>> Load('filename')
    """
    with open(filename, 'rb') as r:
        return pickle.load(r)

class model:
    def __init__(self, name, dataset, **kwargs):
        """
        Base class for Multi-Output Gaussian process models. See subclasses for instantiation.

        Args:
            name (str): Name of the model.
            dataset (mogptk.dataset.DataSet, mogptk.data.Data): DataSet with Data objects for all the channels.
            When a (list or dict of) Data object is passed, it will automatically be converted to a DataSet.
        """
        
        if not isinstance(dataset, DataSet):
            dataset = DataSet(dataset)
        if dataset.get_output_dims() == 0:
            raise Exception("dataset must have at least one channel")
        if len(set(dataset.get_names())) != len(dataset.get_names()):
            raise Exception("all data channels must have unique names")
        if len(set(dataset.get_input_dims())) != 1:
            raise Exception("all data channels must have the same amount of input dimensions")

        for channel in dataset:
            for dim in range(channel.get_input_dims()):
                xran = np.max(channel.X[dim].transformed) - np.min(channel.X[dim].transformed)
                if xran < 1e-3:
                    logger.warning("Very small X range may give problems, it is suggested to scale up your X-axis")
                elif 1e4 < xran:
                    logger.warning("Very large X range may give problems, it is suggested to scale down your X-axis")

        self.name = name
        self.dataset = dataset
    
    def _build(self, kernel):
        """
        Build the model using the given kernel and likelihood. The variational and sparse booleans decide which GPflow model will be used.

        Args:
            kernel (mogptk.kernels.Kernel): Kernel to use.
        """

        x, y = self.dataset._to_kernel()
        self.model = GPR(kernel, x, y)
        self.kernel = self.model.kernel

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
        try:
            os.remove(filename)
        except OSError:
            pass
        with open(filename, 'wb') as w:
            pickle.dump(self, w)

    def train(
        self,
        method='LBFGS',
        tol=1e-6,
        lr=1.0,
        maxiter=500,
        params={},
        verbose=False):
        """
        Trains the model using the kernel and its parameters.

        For different optimizers, see scipy.optimize.minimize.
        It can be bounded by a maximum number of iterations, disp will output final
        optimization information. When using the 'Adam' optimizer, a
        learning_rate can be set.

        Args:
            method (str): Optimizer to use. Defaults to LBFGS.
            tol (float): Tolerance for optimizer. Defaults to 1e-6.
            lr (float): Learning rate for Adam optimizer.
            maxiter (int): Maximum number of iterations. Defaults to 2000.
            params (dict): Additional dictionary with parameters to minimize. 
            verbose (bool): Print verbose output about the state of the optimizer.

        Examples:
            >>> model.train(tol=1e-6, maxiter=10000)
            
            >>> model.train(method='Adam', opt_params={...})
        """
        if verbose:
            training_points = sum([len(channel.get_train_data()[0]) for channel in self.dataset])
            parameters = sum([int(np.prod(param.shape)) for param in self.model.parameters()])
            print('Starting optimization')
            print('‣ Model: {}'.format(self.name))
            print('‣ Channels: {}'.format(len(self.dataset)))
            print('‣ Mixtures: {}'.format(self.Q))
            print('‣ Training points: {}'.format(training_points))
            print('‣ Parameters: {}'.format(parameters))
            print('‣ Initial NLL: {:.3f}'.format(-self.model.log_marginal_likelihood().tolist()))
            inital_time = time.time()

        iters = 0
        try:
            if method.lower() in ('l-bfgs', 'lbfgs'):
                optimizer = torch.optim.LBFGS(self.model.parameters())
                optimizer.step(lambda: self.model.loss())
                iters = optimizer.state_dict()['state'][0]['func_evals']
            elif method.lower() == 'adam':
                optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
                for i in range(maxiter):
                    loss = self.model.loss()
                    optimizer.step()
                iters = maxiter
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

    def predict(self, x=None, plot=False):
        """
        Predict with model.

        Will make a prediction using x as input. If no input value is passed, the prediction will 
        be made with atribute self.X_pred that can be setted with other functions.
        It returns the X, Y_mu, Y_var values per channel.

        Args:
            x_pred (list, dict): Dictionary where keys are channel index and elements numpy arrays with channel inputs.

        returns :
            mu (ndarray): Posterior mean.
            lower (ndarray): Lower confidence interval.
            upper (ndarray): Upper confidence interval.

        Examples:
            >>> model.predict(plot=True)
        """
        if x is not None:
            self.dataset.set_prediction_x(x)

        x = self.dataset._to_kernel_prediction()
        if len(x) == 0:
            raise Exception('no prediction x range set, use x argument or set manually using DataSet.set_prediction_x() or Data.set_prediction_x()')

        mu, var = self.model.predict(x)
        self.dataset._from_kernel_prediction(self.name, mu, var)
        
        if plot:
            self.dataset.plot()

        _, mu, lower, upper = self.dataset.get_prediction(self.name)
        return mu, lower, upper

    def plot_gram_matrix(self, xmin=None, xmax=None, n_points=31, figsize=(10, 10), title='', retmatrix=False):
        """
        Plot the gram matrix of associated kernel.

        The gram matrix is evaluated depending a equaly spaced grid 
        between [xmin_i, xmax_i] for i = 0, ..., n_channels.

        Args:
            xmin (float, list, array): 
            xmax (float, list, array):
            n_points (int): Number of points per channel
            figsize (2-tuple of ints): Figure size.
            title (str): Figure title.
            retmatrix(Bool): if True, return the gram matrix
        Returns:
            fig : Matplotlib figure
            ax : Matplotlib axis

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

        xx = np.zeros((M * n_points, 2))
        xx[:, 0] = np.repeat(np.arange(M), n_points)

        for m in range(M):
            xx[m * n_points: (m + 1) * n_points, 1] = np.linspace(xmin[m], xmax[m], n_points)
            
        K_gram = self.model.kernel.K(xx)
        
        fig, ax = plt.subplots(figsize=figsize)
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
        ax.grid(which='major', alpha=.8, linewidth=1.5, color='k')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(title)

        if retmatrix:
            return fig, ax, K_gram.numpy()
        else:
            return fig, ax

