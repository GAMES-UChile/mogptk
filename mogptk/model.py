import os
import json
import time
import numpy as np
import torch
#import gpflow
#import tensorflow as tf
from .dataset import DataSet
from .kernels import GPR
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

#from gpflow import set_trainable

from tabulate import tabulate

import logging
#logging.getLogger('tensorflow').propagate = False
#logging.getLogger('tensorflow').setLevel(logging.ERROR)

#tf.autograph.set_verbosity(0) # TODO: remove and fix problem

#gpflow.config.set_default_positive_minimum(1e-12)
eps = 1e-20

logger = logging.getLogger('mogptk')

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

    #def get_parameter(self, q, key):
    #    """
    #    Gets a kernel parameter for component 'q' with key the parameter name.

    #    Args:
    #        q (int): Component of kernel.
    #        key (str): Name of component.
    #        
    #    Returns:
    #        val (numpy.ndarray): Value of parameter.

    #    Examples:
    #        >>> val = model.get_parameter(0, 'variance') # for Q=0 get the parameter called 'variance'
    #    """
    #    if hasattr(self.model.kernel, 'kernels'):
    #        if q < 0 or len(self.model.kernel.kernels) <= q:
    #            raise Exception("qth component %d does not exist" % (q,))
    #        kern = self.model.kernel.kernels[q].__dict__
    #    else:
    #        if q != 0:
    #            raise Exception("qth component %d does not exist" % (q,))
    #        kern = self.model.kernel.__dict__
    #    
    #    if key not in kern or not isinstance(kern[key], gpflow.base.Parameter):
    #        raise Exception("parameter name '%s' does not exist for q=%d" % (key, q))
    #
    #    return kern[key].read_value().numpy()

    #def set_parameter(self, q, key, val):
    #    """
    #    Sets a kernel parameter for component 'q' with key the parameter name.

    #    Args:
    #        q (int): Component of kernel.
    #        key (str): Name of component.
    #        val (float, numpy.ndarray): Value of parameter.

    #    Examples:
    #        >>> model.set_parameter(0, 'variance', np.array([5.0, 3.0])) # for Q=0 set the parameter called 'variance'
    #    """
    #    if isinstance(val, (int, float, list)):
    #        val = np.array(val)
    #    if not isinstance(val, np.ndarray):
    #        raise Exception("value %s of type %s is not a number type or ndarray" % (val, type(val)))

    #    if hasattr(self.model.kernel, 'kernels'):
    #        if q < 0 or len(self.model.kernel.kernels) <= q:
    #            raise Exception("qth component %d does not exist" % (q,))
    #        kern = self.model.kernel.kernels[q].__dict__
    #    else:
    #        if q != 0:
    #            raise Exception("qth component %d does not exist" % (q,))
    #        kern = self.model.kernel.__dict__

    #    if key not in kern or not isinstance(kern[key], gpflow.base.Parameter):
    #        raise Exception("parameter name '%s' does not exist for q=%d" % (key, q))

    #    if kern[key].shape != val.shape:
    #        raise Exception("parameter name '%s' must have shape %s and not %s for q=%d" % (key, kern[key].shape, val.shape, q))

    #    # TODO: some parameters can be negative
    #    #for i, v in np.ndenumerate(val):
    #    #    if v < gpflow.config.default_positive_minimum():
    #    #        val[i] = gpflow.config.default_positive_minimum() + eps
    #    kern[key].assign(val)

    #def fix_parameter(self, q, key):
    #    """
    #    Make parameter untrainable (undo with `unfix_parameter`).

    #    Args:
    #        q: (int, list or array-like of ints): components to fix.
    #        key (str): Name of the parameter.

    #    Examples:
    #        >>> model.fix_parameter([0, 1], 'variance')
    #    """

    #    if isinstance(q, int):
    #        q = [q]

    #    if hasattr(self.model.kernel, 'kernels'):
    #        for kernel_i in q:
    #            kernel = self.model.kernel.kernels[kernel_i]
    #            for param_name, param_val in kernel.__dict__.items():
    #                if param_name == key and isinstance(param_val, gpflow.base.Parameter):
    #                    set_trainable(getattr(self.model.kernel.kernels[kernel_i], param_name), False)
    #    else:
    #        for param_name, param_val in self.model.kernel.__dict__.items():
    #            if param_name == key and isinstance(param_val, gpflow.base.Parameter):
    #                set_trainable(getattr(self.model.kernel, param_name), False)

    #def unfix_parameter(self, q, key):
    #    """
    #    Make parameter trainable (that was previously fixed, see `fix_param`).

    #    Args:
    #    q: (int, list or array-like of ints): components to unfix.
    #        key (str): Name of the parameter.

    #    Examples:
    #        >>> model.unfix_parameter('variance')
    #    """

    #    if isinstance(q, int):
    #        q = [q]

    #    if hasattr(self.model.kernel, 'kernels'):
    #         for kernel_i in q:
    #            kernel = self.model.kernel.kernels[kernel_i]
    #            for param_name, param_val in kernel.__dict__.items():
    #                if param_name == key and isinstance(param_val, gpflow.base.Parameter):
    #                    set_trainable(getattr(self.model.kernel.kernels[kernel_i], param_name), False)
    #    else:
    #        for param_name, param_val in self.model.kernel.__dict__.items():
    #            if param_name == key and isinstance(param_val, gpflow.base.Parameter):
    #                set_trainable(getattr(self.model.kernel, param_name), True)

    #def save_parameters(self, filename):
    #    """
    #    Save model parameters to a given file that can then be loaded with `load_parameters()`.

    #    Args:
    #        filename (str): Filename to save to, automatically appends '.params'.

    #    Examples:
    #        >>> model.save_parameters('filename')
    #    """
    #    filename += "." + self.name + ".params"

    #    try:
    #        os.remove(filename)
    #    except OSError:
    #        pass
    #    
    #    class NumpyEncoder(json.JSONEncoder):
    #        def default(self, obj):
    #            if isinstance(obj, np.ndarray):
    #                return obj.tolist()
    #            return json.JSONEncoder.default(self, obj)

    #    data = {
    #        'model': self.__class__.__name__,
    #        'likelihood': self.get_likelihood_parameters(),
    #        'params': self.get_parameters()
    #    }
    #    with open(filename, 'w') as w:
    #        json.dump(data, w, cls=NumpyEncoder)

    #def load_parameters(self, filename):
    #    """
    #    Load model parameters from a given file that was previously saved with `save_parameters()`.

    #    Args:
    #        filename (str): Filename to load from, automatically appends '.params'.

    #    Examples:
    #        >>> model.load_parameters('filename')
    #    """
    #    filename += "." + self.name + ".params"

    #    with open(filename) as r:
    #        data = json.load(r)

    #        if not isinstance(data, dict) or 'model' not in data or 'likelihood' not in data or 'params' not in data:
    #            raise Exception('parameter file has bad format')
    #        if not isinstance(data['params'], list) or not all(isinstance(param, dict) for param in data['params']):
    #            raise Exception('parameter file has bad format')

    #        if data['model'] != self.__class__.__name__:
    #            raise Exception("parameter file uses model '%s' which is different from current model '%s'" % (data['model'], self.__class__.__name__))

    #        cur_params = self.get_parameters()
    #        if len(data['params']) != len(cur_params):
    #            raise Exception("parameter file uses model with %d kernels which is different from current model that uses %d kernels, is the model's Q different?" % (len(data['params']), len(cur_params)))

    #        for key, val in data['likelihood'].items():
    #            self.set_likelihood_parameter(key, val)

    #        for q, param in enumerate(data['params']):
    #            for key, val in param.items():
    #                self.set_parameter(q, key, val)

    def train(
        self,
        method='L-BFGS-B',
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
            method (str): Optimizer to use, if "Adam" is chosen,
                gpflow.training.Adamoptimizer will be used, otherwise the passed scipy
                optimizer is used. Defaults to scipy 'L-BFGS-B'.
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
            print(self.model.log_marginal_likelihood().detach())
            print('Starting optimization')
            print('‣ Model: {}'.format(self.name))
            print('‣ Channels: {}'.format(len(self.dataset)))
            print('‣ Components: {}'.format(self.Q))
            print('‣ Training points: {}'.format(training_points))
            print('‣ Parameters: {}'.format(parameters))
            print('‣ Initial NLL: {:.3f}'.format(-self.model.log_marginal_likelihood().detach().numpy()))
            inital_time = time.time()

        #if method.lower() == 'adam':
        #    opt = tf.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999)
        #    for step in range(maxiter):
        #        opt.minimize(loss, self.model.trainable_variables)
        #else:
        self.model.iters = 0
        optimizer = torch.optim.LBFGS(self.model.parameters(), lr=lr, max_iter=maxiter, tolerance_grad=tol)
        try:
            optimizer.step(self.model.loss)
        except Exception as e:
            print(e)
            return

        if verbose:
            elapsed_time = time.time() - inital_time
            print('\nOptimization finished in {:.2f} minutes'.format(elapsed_time / 60.0))
            print('‣ Function evaluations: {}'.format(self.model.iters))
            print('‣ Final NLL: {:.3f}'.format(-self.model.log_marginal_likelihood().tolist()))

    ################################################################################
    # Predictions ##################################################################
    ################################################################################

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

