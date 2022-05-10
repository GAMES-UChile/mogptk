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

from . import gpr
from .serie import Serie
from .dataset import DataSet
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

    Args:
        variance (float): Variance of the Gaussian likelihood.
        jitter (float): Jitter added before calculating a Cholesky.
    """
    def __init__(self, variance=None, jitter=1e-8):
        self.variance = variance
        self.jitter = jitter

    def _build(self, kernel, x, y, y_err=None, mean=None, name=None):
        variance = self.variance
        if variance is None:
            if y_err is not None:
                variance = (y_err**2).reshape(-1,1)
            else:
                variance = [1.0] * kernel.output_dims
        return gpr.Exact(kernel, x, y, variance=variance, jitter=self.jitter, mean=mean, name=name)

class Snelson:
    """
    Inference using Snelson and Ghahramani 2005 for Gaussian process regression.

    Args:
        inducing_points (int,list): Number of inducing points or the locations of the inducing points.
        init_inducing_points (str): Method for initialization of inducing points, can be `grid`, `random`, or `density`.
        variance (float): Variance of the Gaussian likelihood.
        jitter (float): Jitter added before calculating a Cholesky.
    """
    def __init__(self, inducing_points=10, init_inducing_points='grid', variance=None, jitter=1e-6):
        self.inducing_points = inducing_points
        self.init_inducing_points = init_inducing_points
        self.variance = variance
        self.jitter = jitter

    def _build(self, kernel, x, y, y_err=None, mean=None, name=None):
        if variance is None:
            variance = [1.0] * kernel.output_dims
        return gpr.Snelson(kernel, x, y, Z=self.inducing_points, Z_init=self.init_inducing_points, variance=self.variance, jitter=self.jitter, mean=mean, name=name)

class OpperArchambeau:
    """
    Inference using Opper and Archambeau 2009 for Gaussian process regression.

    Args:
        likelihood (gpr.Likelihood): Likelihood $p(y|f)$.
        jitter (float): Jitter added before calculating a Cholesky.
    """
    def __init__(self, likelihood=gpr.GaussianLikelihood(variance=1.0), jitter=1e-6):
        self.likelihood = likelihood
        self.jitter = jitter

    def _build(self, kernel, x, y, y_err=None, mean=None, name=None):
        return gpr.OpperArchambeau(kernel, x, y, likelihood=likelihood, jitter=self.jitter, mean=mean, name=name)

class Titsias:
    """
    Inference using Titsias 2009 for Gaussian process regression.

    Args:
        inducing_points (int,list): Number of inducing points or the locations of the inducing points.
        init_inducing_points (str): Method for initialization of inducing points, can be `grid`, `random`, or `density`.
        variance (float): Variance of the Gaussian likelihood.
        jitter (float): Jitter added before calculating a Cholesky.
    """
    def __init__(self, inducing_points=10, init_inducing_points='grid', variance=1.0, jitter=1e-6):
        self.inducing_points = inducing_points
        self.init_inducing_points = init_inducing_points
        self.variance = variance
        self.jitter = jitter

    def _build(self, kernel, x, y, y_err=None, mean=None, name=None):
        return gpr.Titsias(kernel, x, y, Z=self.inducing_points, Z_init=self.init_inducing_points, variance=self.variance, jitter=self.jitter, mean=mean, name=name)

class Hensman:
    """
    Inference using Hensman 2015 for Gaussian process regression.

    Args:
        inducing_points (int,list): Number of inducing points or the locations of the inducing points. By default the non-sparse Hensman model is used.
        init_inducing_points (str): Method for initialization of inducing points, can be `grid`, `random`, or `density`.
        likelihood (gpr.Likelihood): Likelihood $p(y|f)$.
        jitter (float): Jitter added before calculating a Cholesky.
    """
    def __init__(self, inducing_points=None, init_inducing_points='grid', likelihood=gpr.GaussianLikelihood(variance=1.0), jitter=1e-6):
        self.inducing_points = inducing_points
        self.init_inducing_points = init_inducing_points
        self.likelihood = likelihood
        self.jitter = jitter

    def _build(self, kernel, x, y, y_err=None, mean=None, name=None):
        if self.inducing_points is None:
            return gpr.Hensman(kernel, x, y, likelihood=self.likelihood, jitter=self.jitter, mean=mean, name=name)
        return gpr.SparseHensman(kernel, x, y, Z=self.inducing_points, Z_init=self.init_inducing_points, likelihood=self.likelihood, jitter=self.jitter, mean=mean, name=name)

class Model:
    def __init__(self, dataset, kernel, inference=Exact(), mean=None, name=None, rescale_x=False):
        """
        Model is the base class for multi-output Gaussian process models.

        Args:
            dataset (mogptk.dataset.DataSet, mogptk.data.Data): `DataSet` with `Data` objects for all the channels. When a (list or dict of) `Data` object is passed, it will automatically be converted to a `DataSet`.
            kernel (mogptk.gpr.kernel.Kernel): The kernel class.
            model: Gaussian process model to use, such as `mogptk.model.Exact`.
            mean (mogptk.gpr.mean.Mean): The mean class.
            name (str): Name of the model.
            rescale_x (bool): Rescale the X axis to [0,1000] to help training.

        Atributes:
            dataset: The associated mogptk.dataset.DataSet.
            gpr: The mogptk.gpr.model.Model.
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

        X = [[x[channel.mask] for x in channel.X] for channel in self.dataset]
        Y = [np.array(channel.Y[channel.mask]) for channel in self.dataset]
        x, y = self._to_kernel_format(X, Y)

        y_err = None
        if all(channel.Y_err is not None for channel in self.dataset):
            # TODO: doesn't transform...
            Y_err = [np.array(channel.Y_err[channel.mask]) for channel in self.dataset]
            Y_err_lower = [self.dataset[j].Y.transform(Y[j] - Y_err[j], X[j]) for j in range(len(self.dataset))]
            Y_err_upper = [self.dataset[j].Y.transform(Y[j] + Y_err[j], X[j]) for j in range(len(self.dataset))]
            y_err_lower = np.concatenate(Y_err_lower, axis=0)
            y_err_upper = np.concatenate(Y_err_upper, axis=0)
            y_err = (y_err_upper-y_err_lower)/2.0 # TODO: strictly incorrect: takes average error after transformation
        self.gpr = inference._build(kernel, x, y, y_err, mean, name)

    ################################################################

    def print_parameters(self):
        """
        Print the parameters of the model in a table.

        Examples:
            >>> model.print_parameters()
        """
        self.gpr.print_parameters()

    def get_parameters(self):
        """
        Returns all parameters of the kernel.

        Returns:
            list: mogptk.gpr.parameter.Parameter

        Examples:
            >>> params = model.get_parameters()
        """
        return self.gpr.get_parameters()

    def copy_parameters(self, other):
        """
        Copy the kernel parameters from another model.
        """
        if not isinstance(other, Model):
            raise ValueError("other must be of type Model")

        self.gpr.kernel.copy_parameters(other.kernel)

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
        return self.gpr.log_marginal_likelihood().detach().cpu().item()

    def loss(self):
        """
        Returns the loss of the kernel and its data and parameters.

        Returns:
            float: The current loss.

        Examples:
            >>> model.loss()
        """
        return self.gpr.loss().detach().cpu().item()

    def error(self, method='MAE', use_all_data=False):
        """
        Returns the error of the kernel prediction with the removed data points in the data set.

        Args:
            method (str): Error calculation method, such as MAE, MAPE, sMAPE, MSE, or RMSE.

        Returns:
            float: The current error.

        Examples:
            >>> model.error()
        """
        if use_all_data:
            X, Y_true = self.dataset.get_data()
        else:
            X, Y_true = self.dataset.get_test_data()
        x, y_true  = self._to_kernel_format(X, Y_true)
        y_pred, _ = self.gpr.predict(x, predict_y=False)
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
            plot (bool): Plot the loss and, if error is data set, the error of the test data points.
            **kwargs (dict): Additional dictionary of parameters passed to the PyTorch optimizer. 

        Returns:
            numpy.ndarray: Losses for all iterations.
            numpy.ndarray: Errors for all iterations. Only if `error` is set, otherwise zero.

        Examples:
            >>> model.train()
            
            >>> model.train(method='lbfgs', tolerance_grad=1e-10, tolerance_change=1e-12)
            
            >>> model.train(method='adam', lr=0.5)
        """
        error_use_all_data = False
        if error is not None and all(not channel.has_test_data() for channel in self.dataset):
            error_use_all_data = True

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
            parameters = sum([p.num_parameters if p.trainable else 0 for p in self.gpr.get_parameters()])
            print('\nStarting optimization using', method)
            print('‣ Model: {}'.format(self.name))
            print('‣ Channels: {}'.format(len(self.dataset)))
            if hasattr(self, 'Q'):
                print('‣ Mixtures: {}'.format(self.Q))
            print('‣ Training points: {}'.format(training_points))
            print('‣ Parameters: {}'.format(parameters))
            print('‣ Initial loss: {:.3g}'.format(self.loss()))
            if error is not None:
                print('‣ Initial error: {:.3g}'.format(self.error(error, error_use_all_data)))

        losses = np.empty((iters+1,))
        errors = np.zeros((iters+1,))

        inital_time = time.time()
        sys.__stdout__.write("\nStart %s:\n" % (method,))
        if method == 'LBFGS':
            if 'lr' not in kwargs:
                kwargs['lr'] = 0.1
            if not 'max_iter' in kwargs:
                kwargs['max_iter'] = iters
                iters = 0
            optimizer = torch.optim.LBFGS(self.gpr.parameters(), **kwargs)

            def loss():
                i = int(optimizer.state_dict()['state'][0]['func_evals'])
                elapsed_time = time.time() - inital_time
                losses[i] = self.loss()
                if error is not None:
                    errors[i] = self.error(error, error_use_all_data)
                    if i % (kwargs['max_iter']/100) == 0:
                        sys.__stdout__.write("% 5d/%d %s  loss=%10g  error=%10g\n" % (i, kwargs['max_iter'], _format_time(elapsed_time), losses[i], errors[i]))
                elif i % (kwargs['max_iter']/100) == 0:
                    sys.__stdout__.write("% 5d/%d %s  loss=%10g\n" % (i, kwargs['max_iter'], _format_time(elapsed_time), losses[i]))
                return losses[i]
            optimizer.step(loss)
            iters = int(optimizer.state_dict()['state'][0]['func_evals'])
        else:
            if method == 'Adam':
                if 'lr' not in kwargs:
                    kwargs['lr'] = 0.1
                optimizer = torch.optim.Adam(self.gpr.parameters(), **kwargs)
            elif method == 'SGD':
                optimizer = torch.optim.SGD(self.gpr.parameters(), **kwargs)
            elif method == 'AdaGrad':
                optimizer = torch.optim.Adagrad(self.gpr.parameters(), **kwargs)
            else:
                print("Unknown optimizer:", method)

            for i in range(iters):
                elapsed_time = time.time() - inital_time
                losses[i] = self.loss()
                if error is not None:
                    errors[i] = self.error(error, error_use_all_data)
                    if i % (iters/100) == 0:
                        sys.__stdout__.write("% 5d/%d %s  loss=%10g  error=%10g\n" % (i, iters, _format_time(elapsed_time), losses[i], errors[i]))
                elif i % (iters/100) == 0:
                    sys.__stdout__.write("% 5d/%d %s  loss=%10g\n" % (i, iters, _format_time(elapsed_time), losses[i]))
                optimizer.step()
        losses[iters] = self.loss()
        elapsed_time = time.time() - inital_time
        if error is not None:
            errors[iters] = self.error(error, error_use_all_data)
            sys.__stdout__.write("% 5d/%d %s  loss=%10g  error=%10g\n" % (iters, iters, _format_time(elapsed_time), losses[iters], errors[iters]))
        else:
            sys.__stdout__.write("% 5d/%d %s  loss=%10g\n" % (iters, iters, _format_time(elapsed_time), losses[iters]))
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
        X_orig = X
        X = X.copy()
        for j, channel_x in enumerate(X):
            if channel_x is None or len(channel_x) == 0:
                X[j] = np.empty((0, input_dims))
                continue

            input_dims = self.dataset.get_input_dims()[j]
            if isinstance(channel_x, np.ndarray):
                if channel_x.ndim == 1:
                    channel_x = channel_x.reshape(-1, 1)
                if channel_x.ndim != 2 or channel_x.shape[1] != input_dims:
                    raise ValueError("X must be of shape (n,input_dims) or a list [(n,)] * input_dims for each channel")
                channel_x = [channel_x[:,i] for i in range(input_dims)]
            elif not isinstance(channel_x, list):
                raise ValueError("X must be a list of lists or numpy.ndarrays")
            if not all(isinstance(x, np.ndarray) for x in channel_x) or len(channel_x) != input_dims:
                raise ValueError("X must be of shape (n,input_dims) or a list [(n,)] * input_dims for each channel")
            X[j] = np.array([self.dataset[j].X[i].transform(channel_x[i]) for i in range(input_dims)]).T

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
        Y = Y.copy()
        for j, channel_y in enumerate(Y):
            if channel_y.ndim != 1:
                raise ValueError("Y must be a list of shape (n,) for each channel")
            if channel_y.shape[0] != X[j].shape[0]:
                raise ValueError("Y must have the same number of data points per channel as X")
            Y[j] = self.dataset[j].Y.transform(channel_y, X_orig[j])
        if len(Y) == 0:
            y = np.array([])
        else:
            y = np.concatenate(Y, axis=0).reshape(-1, 1)
        return x, y

    def predict(self, X=None, sigma=None, q=[0.025, 0.975], transformed=False, predict_y=True):
        """
        Predict using the prediction range of the data set and save the prediction in that data set. Otherwise, if `X` is passed, use that as the prediction range and return the prediction instead of saving it.

        Args:
            X (list, dict): Dictionary where keys are channel index and elements numpy arrays with channel inputs. If passed, results will be returned and not saved in the data set for later retrieval.
            q (list of float): The quantiles for the confidence interval's lower and upper ends.
            transformed (boolean): Return transformed data as used for training.

        Returns:
            numpy.ndarray: Y mean prediction of shape (n,) for each channel.
            numpy.ndarray: Y lower prediction of uncertainty interval of shape (n,) for each channel.
            numpy.ndarray: Y upper prediction of uncertainty interval of shape (n,) for each channel.

        Examples:
            >>> model.predict(X)
        """
        save = X is None
        if save and transformed:
            raise ValueError('must pass an X range explicitly in order to return transformed data')
        if save:
            X = self.dataset.get_prediction_x()
        else:
            X = self.dataset._format_prediction_x(X)
        x = self._to_kernel_format(X)

        mu, var = self.gpr.predict(x, predict_y=predict_y, tensor=True)
        # TODO: quantiles
        #if predict_y:
            #lower = self.gpr.quantile(q[0], mu, var)
            #upper = self.gpr.quantile(q[1], mu, var)
        if sigma is not None:
            lower = mu - sigma*torch.sqrt(var)
            upper = mu + sigma*torch.sqrt(var)
        else:
            ql = torch.tensor(q[0], device=gpr.config.device, dtype=gpr.config.dtype)
            qu = torch.tensor(q[1], device=gpr.config.device, dtype=gpr.config.dtype)
            lower = mu + torch.sqrt(var)*np.sqrt(2)*torch.special.erfinv(2.0*ql - 1.0)
            upper = mu + torch.sqrt(var)*np.sqrt(2)*torch.special.erfinv(2.0*qu - 1.0)

        mu = mu.cpu().numpy()
        var = var.cpu().numpy()
        lower = lower.cpu().numpy()
        upper = upper.cpu().numpy()

        i = 0
        Mu = []
        Var = []
        Lower = []
        Upper = []
        for j in range(self.dataset.get_output_dims()):
            N = X[j][0].shape[0]
            Mu.append(np.squeeze(mu[i:i+N]))
            Var.append(np.squeeze(var[i:i+N]))
            Lower.append(np.squeeze(lower[i:i+N]))
            Upper.append(np.squeeze(upper[i:i+N]))
            i += N

        if save:
            for j in range(self.dataset.get_output_dims()):
                self.dataset[j].Y_mu_pred[self.name] = Mu[j]
                self.dataset[j].Y_var_pred[self.name] = Var[j]

        if not transformed:
            for j in range(self.dataset.get_output_dims()):
                Mu[j] = self.dataset[j].Y.detransform(Mu[j], X[j])
                Lower[j] = self.dataset[j].Y.detransform(Lower[j], X[j])
                Upper[j] = self.dataset[j].Y.detransform(Upper[j], X[j])
        return Mu, Lower, Upper

    def K(self, X1, X2=None):
        """
        Evaluate the kernel at K(X1,X2).

        Args:
            X1 (list, dict): Dictionary where keys are channel index and elements numpy arrays with channel inputs.
            X2 (list, dict): Same as X1 if None.

        Returns:
            numpy.ndarray: kernel evaluated at X1 and X2 of shape (n1,n2).

        Examples:
            >>> channel0 = np.array(['1987-05-20', '1987-05-21'])
            >>> channel1 = np.array([[2.5, 534.6], [3.5, 898.22], [4.5, 566.98]])
            >>> model.K([channel0,channel1])
        """
        x1 = self._to_kernel_format(X1)
        if X2 is None:
            return self.gpr.K(x1)
        else:
            x2 = self._to_kernel_format(X2)
            return self.gpr.K(x1, x2)

    def sample(self, X=None, n=None, predict_y=True, transformed=False):
        """
        Sample n times from the kernel at input X .

        Args:
            X (list, dict): Dictionary where keys are channel index and elements numpy arrays with channel inputs.
            n (int): Number of samples.
            transformed (boolean): Return transformed data as used for training.

        Returns:
            list: samples of shape len(X) for each channel if n is given.
            numpy.ndarray: sample of shape len(X) for each channel if n is None.

        Examples:
            >>> model.sample(n=10)
        """
        if X is None:
            X = self.dataset.get_prediction_x()
        else:
            X = self.dataset._format_prediction_x(X)
        x = self._to_kernel_format(X)

        samples = self.gpr.sample(Z=x, n=n, predict_y=predict_y)

        i = 0
        Samples = []
        for j in range(self.dataset.get_output_dims()):
            N = X[j][0].shape[0]
            if n is None:
                sample = np.squeeze(samples[i:i+N])
                if not transformed:
                    sample = self.dataset[j].Y.detransform(sample, X[j])
                Samples.append(sample)
            else:
                ss = []
                for k in range(n):
                    sample = np.squeeze(samples[i:i+N,k])
                    if not transformed:
                        sample = self.dataset[j].Y.detransform(sample, X[j])
                    ss.append(sample)
                Samples.append(ss)
            i += N
        if self.dataset.get_output_dims() == 1:
            return Samples[0]
        return Samples

    def plot_losses(self, title=None, figsize=None, legend=True, errors=True):
        """
        Plot the losses and errors during training. In order to display the errors, make sure to set the error parameter when training.

        Args:
            title (str): Figure title.
            figsize (tuple): Figure size.
            legend (boolean): Show the legend.
            errors (boolean): Show the errors.

        Returns:
            figure: Matplotlib figure.
            axis: Matplotlib axis.
        """
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
        return fig, ax

    def plot_prediction(self, X=None, title=None, figsize=None, legend=True, transformed=False, predict_y=True):
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
        if X is not None:
            self.dataset.set_prediction_x(X)
            self.predict()
        elif not self.name in self.dataset[0].Y_mu_pred:
            self.predict(predict_y=predict_y)
        return self.dataset.plot(pred=self.name, title=title, figsize=figsize, legend=legend, transformed=transformed)

    def plot_gram(self, start=None, end=None, n=31, title=None, figsize=(12,12)):
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
        if not all(channel.get_input_dims() == 1 for channel in self.dataset):
            raise ValueError("cannot plot for more than one input dimension")

        if start is None:
            start = [channel.X[0].transformed.min() for channel in self.dataset]
        if end is None:
            end = [channel.X[0].transformed.max() for channel in self.dataset]

        output_dims = len(self.dataset)
        if not isinstance(start, (list, np.ndarray)):
            start = [start] * output_dims
        if not isinstance(end, (list, np.ndarray)):
            end = [end] * output_dims

        X = np.zeros((output_dims*n, 2))
        X[:,0] = np.repeat(np.arange(output_dims), n)
        for j in range(output_dims):
            if n== 1:
                X[j*n:(j+1)*n,1] = np.array((start[j]+end[j])/2.0)
            else:
                X[j*n:(j+1)*n,1] = np.linspace(start[j], end[j], n)
        k = self.gpr.K(X)
            
        fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
        if title is not None:
            fig.suptitle(title, fontsize=18)

        color_range = np.abs(k).max()
        norm = matplotlib.colors.Normalize(vmin=-color_range, vmax=color_range)
        im = ax.matshow(k, cmap='coolwarm', norm=norm)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.3)
        fig.colorbar(im, cax=cax)

        # Major ticks every 20, minor ticks every 5
        major_ticks = np.arange(-0.5, output_dims*n, n)
        minor_ticks = np.arange(-0.5, output_dims*n, 2)

        ax.set_xticks(major_ticks)
        ax.set_yticks(major_ticks)
        ax.grid(which='major', lw=1.5, c='k')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='both', length=0)
        return fig, ax

    def plot_kernel(self, dist=None, n=101, title=None, figsize=(12,12)):
        """
        Plot the kernel matrix at a range of data point distances for each channel for stationary kernels.

        Args:
            dist (list): Maximum distance for every channel.
            n (int): Number of points per channel.
            title (str): Figure title.
            figsize (tuple): Figure size.

        Returns:
            figure: Matplotlib figure.
            axis: Matplotlib axis.
        """
        if not all(channel.get_input_dims() == 1 for channel in self.dataset):
            raise ValueError("cannot plot for more than one input dimension")

        if dist is None:
            dist = [(channel.X[0].transformed.max()-channel.X[0].transformed.min())/4.0 for channel in self.dataset]

        output_dims = len(self.dataset)
        if not isinstance(dist, (list, np.ndarray)):
            dist = [dist] * output_dims

        fig, ax = plt.subplots(output_dims, output_dims, figsize=figsize, constrained_layout=True, squeeze=False, sharex=True)
        if title is not None:
            fig.suptitle(title, fontsize=18)

        channel = np.ones((n,1))
        for j in range(output_dims):
            tau = np.linspace(-dist[j], dist[j], num=n).reshape(-1,1)
            X1 = np.array([[j,0.0]])
            for i in range(output_dims):
                if j < i:
                    ax[j,i].set_axis_off()
                    continue

                X0 = np.concatenate((i*channel,tau), axis=1)
                k = self.gpr.K(X0,X1)
                ax[j,i].plot(tau, k, color='k')
                ax[j,i].set_yticks([])
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

def _format_time(s):
    return "%3d:%02d:%02d" % (int(s/3600), int((s%3600)/60), int(s%60))
