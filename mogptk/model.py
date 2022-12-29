import os
import time
import math
import pickle
import inspect
import numpy as np
import torch
import logging
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable

from . import gpr
from .dataset import DataSet
from .util import *

logger = logging.getLogger('mogptk')

class Kernels(dict):
    __getattr__ = dict.get

kernels = Kernels({
    'White': gpr.WhiteKernel,
    'Constant': gpr.ConstantKernel,
    'Linear': gpr.LinearKernel,
    'Polynomial': gpr.PolynomialKernel,
    'Function': gpr.FunctionKernel,
    'Exponential': gpr.ExponentialKernel,
    'Exp': gpr.ExponentialKernel,
    'SquaredExponential': gpr.SquaredExponentialKernel,
    'SqExp': gpr.SquaredExponentialKernel,
    'SE': gpr.SquaredExponentialKernel,
    'RBF': gpr.SquaredExponentialKernel,
    'RationalQuadratic': gpr.RationalQuadraticKernel,
    'RQ': gpr.RationalQuadraticKernel,
    'Periodic': gpr.PeriodicKernel,
    'ExpSineSquared': gpr.PeriodicKernel,
    'LocallyPeriodic': gpr.LocallyPeriodicKernel,
    'Cosine': gpr.CosineKernel,
    'Sinc': gpr.SincKernel,
    'Spectral': gpr.SpectralKernel,
    'SpectralMixture': gpr.SpectralMixtureKernel,
    'Matern': gpr.MaternKernel,
    'IndependentMultiOutput': gpr.IndependentMultiOutputKernel,
    'IMO': gpr.IndependentMultiOutputKernel,
    'MultiOutputSpectral': gpr.MultiOutputSpectralKernel,
    'MultiOutputSpectralMixture': gpr.MultiOutputSpectralMixtureKernel,
    'MOSM': gpr.MultiOutputSpectralMixtureKernel,
    'UncoupledMultiOutputSpectral': gpr.UncoupledMultiOutputSpectralKernel,
    'uMOS': gpr.UncoupledMultiOutputSpectralKernel,
    'MultiOutputHarmonizableSpectral': gpr.MultiOutputHarmonizableSpectralKernel,
    'MOHS': gpr.MultiOutputHarmonizableSpectralKernel,
    'CrossSpectral': gpr.CrossSpectralKernel,
    'LinearModelOfCoregionalization': gpr.LinearModelOfCoregionalizationKernel,
    'LMC': gpr.LinearModelOfCoregionalizationKernel,
    'GaussianConvolutionProcess': gpr.GaussianConvolutionProcessKernel,
    'CONV': gpr.GaussianConvolutionProcessKernel,
    'GCP': gpr.GaussianConvolutionProcessKernel,
})

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
    def __init__(self, variance=None, data_variance=None, jitter=1e-8):
        self.variance = variance
        self.data_variance = data_variance
        self.jitter = jitter

    def _build(self, kernel, x, y, y_err=None, mean=None, name=None):
        variance = self.variance
        if variance is None:
            if kernel.output_dims is not None:
                variance = [1.0] * kernel.output_dims
            else:
                variance = 1.0
        data_variance = self.data_variance
        if data_variance is None and y_err is not None:
            data_variance = y_err**2
        model = gpr.Exact(kernel, x, y, variance=variance, data_variance=data_variance, jitter=self.jitter, mean=mean, name=name)
        return model

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
    def __init__(self, likelihood=gpr.GaussianLikelihood(1.0), jitter=1e-6):
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
    def __init__(self, inducing_points=None, init_inducing_points='grid', likelihood=gpr.GaussianLikelihood(1.0), jitter=1e-6):
        self.inducing_points = inducing_points
        self.init_inducing_points = init_inducing_points
        self.likelihood = likelihood
        self.jitter = jitter

    def _build(self, kernel, x, y, y_err=None, mean=None, name=None):
        if self.inducing_points is None:
            return gpr.Hensman(kernel, x, y, likelihood=self.likelihood, jitter=self.jitter, mean=mean, name=name)
        return gpr.SparseHensman(kernel, x, y, Z=self.inducing_points, Z_init=self.init_inducing_points, likelihood=self.likelihood, jitter=self.jitter, mean=mean, name=name)

class Model:
    def __init__(self, dataset, kernel, inference=Exact(), mean=None, name=None):
        """
        Model is the base class for multi-output Gaussian process models.

        Args:
            dataset (mogptk.dataset.DataSet, mogptk.data.Data): `DataSet` with `Data` objects for all the channels. When a (list or dict of) `Data` object is passed, it will automatically be converted to a `DataSet`.
            kernel (mogptk.gpr.kernel.Kernel): The kernel class.
            inference: Gaussian process inference model to use, such as `mogptk.Exact`.
            mean (mogptk.gpr.mean.Mean): The mean class.
            name (str): Name of the model.

        Attributes:
            dataset (mogptk.dataset.DataSet): Dataset.
            gpr (mogptk.gpr.model.Model): GPR model.
            times (numpy.ndarray): Training times of shape (iters,).
            losses (numpy.ndarray): Losses of shape (iters,).
            errors (numpy.ndarray): Errors of shape (iters,).
        """
        
        if not isinstance(dataset, DataSet):
            dataset = DataSet(dataset)
        if dataset.get_output_dims() == 0:
            raise ValueError("dataset must have at least one channel")
        names = [name for name in dataset.get_names() if name is not None]
        if len(set(names)) != len(names):
            raise ValueError("all data channels must have unique names")

        #for j, channel in enumerate(dataset):
        #    for dim in range(channel.get_input_dims()):
        #        xran = np.max(channel.X[:,dim]) - np.min(channel.X[:,dim])
        #        if xran < 1e-3:
        #            logger.warning("Very small X range may give problems, it is suggested to scale up your X axis for channel %d" % j)
        #        elif 1e4 < xran:
        #            logger.warning("Very large X range may give problems, it is suggested to scale down your X axis for channel %d" % j)

        self.name = name
        self.dataset = dataset
        self.is_multioutput = kernel.output_dims is not None

        X, Y = self.dataset.get_train_data()
        x, y = self._to_kernel_format(X, Y)

        y_err = None
        if all(channel.Y_err is not None for channel in self.dataset):
            Y_err = [channel.Y_err[channel.mask] for channel in self.dataset]
            Y_err_lower = [self.dataset[j].Y_transformer.forward(Y[j] - Y_err[j], X[j]) for j in range(len(self.dataset))]
            Y_err_upper = [self.dataset[j].Y_transformer.forward(Y[j] + Y_err[j], X[j]) for j in range(len(self.dataset))]
            y_err_lower = np.concatenate(Y_err_lower, axis=0)
            y_err_upper = np.concatenate(Y_err_upper, axis=0)
            y_err = (y_err_upper-y_err_lower)/2.0 # TODO: strictly incorrect: takes average error after transformation
        self.gpr = inference._build(kernel, x, y, y_err, mean, name)

        self.iters = 0
        self.times = np.zeros(0)
        self.losses = np.zeros(0)
        self.errors = np.zeros(0)

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

        self.gpr.kernel.copy_parameters(other.gpr.kernel)

    def num_parameters(self):
        """
        Returns the number of trainable parameters.

        Returns:
            int: Number of parameters.

        Examples:
            >>> n = model.num_parameters()
        """
        return sum([p.num_parameters if p.train else 0 for p in self.gpr.get_parameters()])

    def num_training_points(self):
        """
        Returns the number of training data points.

        Returns:
            int: Number of data points.

        Examples:
            >>> n = model.num_training_points()
        """
        return sum([len(channel.get_train_data()[1]) for channel in self.dataset])

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
        Returns the log marginal likelihood of the kernel and its data and parameters. When using the exact model the calculation of the log marginal likelihood is tractable and thus exact. For other models this is an approximation of the real log marginal likelihood.

        Returns:
            float: The current log marginal likelihood.

        Examples:
            >>> model.log_marginal_likelihood()
        """
        return self.gpr.log_marginal_likelihood().detach().cpu().item()

    def BIC(self):
        """
        Returns the Bayesian information criterion.

        Returns:
            float: BIC.

        Examples:
            >>> model.BIC()
        """
        return self.num_parameters()*np.log(self.num_training_points()) - 2.0*self.log_marginal_likelihood()

    def AIC(self):
        """
        Returns the Akaike information criterion.

        Returns:
            float: AIC.

        Examples:
            >>> model.AIC()
        """
        return 2.0*self.num_parameters() - 2.0*self.log_marginal_likelihood()

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
            method (str,function): Error calculation method, such as MAE, MAPE, sMAPE, MSE, or RMSE. When a function is given, it should have parameters (y_true,y_pred) or (y_true,y_pred,model).

        Returns:
            float: The current error.

        Examples:
            >>> model.error()
        """

        if callable(method) and len(inspect.signature(method).parameters) == 1:
            return method(self)

        # get data
        if use_all_data or not any(self.dataset.has_test_data()):
            X, Y_true = self.dataset.get_data()
        else:
            X, Y_true = self.dataset.get_test_data()

        # predict
        x  = self._to_kernel_format(X)
        y_pred, _ = self.gpr.predict(x, predict_y=False)

        # transform to original
        i = 0
        Y_pred = []
        for j in range(self.dataset.get_output_dims()):
            N = X[j].shape[0]
            Y_pred.append(self.dataset[j].Y_transformer.backward(np.squeeze(y_pred[i:i+N]), X[j]))
            i += N

        # flatten
        y_true = np.concatenate(Y_true)
        y_pred = np.concatenate(Y_pred)

        if callable(method):
            return method(y_true, y_pred)
        elif method.lower() == 'mae':
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
            raise ValueError("valid error calculation methods are MAE, MAPE, sMAPE, MSE, and RMSE")

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
            error (str,function): Calculate prediction error for each iteration by the given method, such as MAE, MAPE, sMAPE, MSE, or RMSE. When a function is given, it should have parameters (y_true,y_pred) or (y_true,y_pred,model).
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

        if callable(error):
            if len(inspect.signature(error).parameters) == 1:
                e = error(self)
            else:
                e = error(np.zeros((1,1)), np.zeros((1,1)))
            if not isinstance(e, float) and (not isinstance(e, np.ndarray) or e.size != 1):
                raise ValueError("error function must return a float")

        if method.lower() in ('l-bfgs', 'lbfgs', 'l-bfgs-b', 'lbfgsb'):
            method = 'LBFGS'
        elif method.lower() == 'adam':
            method = 'Adam'
        elif method.lower() == 'sgd':
            method = 'SGD'
        elif method.lower() == 'adagrad':
            method = 'AdaGrad'
        else:
            raise ValueError('optimizer must be LBFGS, Adam, SGD, or AdaGrad')

        if verbose:
            print('\nStarting optimization using', method)
            if self.name is not None:
                print('‣ Model: %s' % self.name)
            print('‣ Channels: %d' % len(self.dataset))
            print('‣ Parameters: %d' % self.num_parameters())
            print('‣ Training points: %d' % self.num_training_points())
            print('‣ Initial loss: %6g' % self.loss())
            if error is not None:
                print('‣ Initial error: %6g' % self.error(error, error_use_all_data))

        iter_offset = 0
        times = np.zeros((iters+1,))
        losses = np.zeros((iters+1,))
        errors = np.zeros((iters+1,))
        if self.times.shape[0] != 0:
            iter_offset = self.times.shape[0]-1
            times = np.concatenate((self.times[:-1],times))
            losses = np.concatenate((self.losses[:-1],losses))
            errors = np.concatenate((self.errors[:-1],errors))
        initial_time = time.time()

        iters_len = int(math.log10(iter_offset+iters)) + 1
        def progress(i, loss):
            elapsed_time = time.time() - initial_time
            write = verbose and i % max(1,iters/100) == 0
            i += iter_offset
            times[i] = elapsed_time
            losses[i] = loss
            if error is not None:
                errors[i] = float(self.error(error, error_use_all_data))
                if write:
                    print("  %*d/%*d %s  loss=%12g  error=%12g" % (iters_len, i, iters_len, iter_offset+iters, _format_time(elapsed_time), losses[i], errors[i]))
            elif write:
                print("  %*d/%*d %s  loss=%12g" % (iters_len, i, iters_len, iter_offset+iters, _format_time(elapsed_time), losses[i]))

        if verbose:
            print("\nStart %s:" % (method,))
        if method == 'LBFGS':
            if not 'max_iter' in kwargs:
                kwargs['max_iter'] = iters
            else:
                iters = kwargs['max_iter']
            optimizer = torch.optim.LBFGS(self.gpr.parameters(), **kwargs)

            def loss():
                i = int(optimizer.state_dict()['state'][0]['func_evals'])
                loss = self.loss()
                progress(i, loss)
                return loss
            optimizer.step(loss)
            iters = int(optimizer.state_dict()['state'][0]['func_evals'])
        else:
            if method == 'Adam':
                optimizer = torch.optim.Adam(self.gpr.parameters(), **kwargs)
            elif method == 'SGD':
                optimizer = torch.optim.SGD(self.gpr.parameters(), **kwargs)
            elif method == 'AdaGrad':
                optimizer = torch.optim.Adagrad(self.gpr.parameters(), **kwargs)

            for i in range(iters):
                progress(i, self.loss())
                optimizer.step()
        progress(iters, self.loss())

        if verbose:
            elapsed_time = time.time() - initial_time
            print("Finished")
            print('\nOptimization finished in %s' % _format_duration(elapsed_time))
            print('‣ Iterations: %d' % iters)
            print('‣ Final loss: %6g'% losses[iter_offset+iters])
            if error is not None:
                print('‣ Final error: %6g' % errors[iter_offset+iters])

        self.iters = iter_offset+iters
        self.times = times[:iter_offset+iters+1]
        self.losses = losses[:iter_offset+iters+1]
        if error is not None:
            self.errors = errors[:iter_offset+iters+1]
        if plot:
            self.plot_losses()
        return losses, errors

    ################################################################################
    # Predictions ##################################################################
    ################################################################################

    def _to_kernel_format(self, X, Y=None):
        """
        Return the data vectors in the format used by the kernels. If Y is not passed, than only X data is returned.

        Returns:
            numpy.ndarray: X data of shape (data_points,input_dims). If the kernel is multi output, an additional input dimension is prepended with the channel indices.
            numpy.ndarray: Y data of shape (data_points,1).
            numpy.ndarray: Original but normalized X data. Only if no Y is passed.
        """
        x = np.concatenate(X, axis=0)
        if self.is_multioutput:
            chan = [j * np.ones(len(X[j])) for j in range(len(X))]
            chan = np.concatenate(chan).reshape(-1, 1)
            x = np.concatenate([chan, x], axis=1)
        if Y is None:
            return x

        Y = list(Y) # shallow copy
        for j, channel_y in enumerate(Y):
            Y[j] = self.dataset[j].Y_transformer.forward(Y[j], X[j])
        y = np.concatenate(Y, axis=0).reshape(-1, 1)
        return x, y

    def predict(self, X=None, sigma=2, predict_y=True, transformed=False):
        """
        Predict using the prediction range of the data set and save the prediction in that data set. Otherwise, if `X` is passed, use that as the prediction range and return the prediction instead of saving it.

        Args:
            X (list, dict): Array of shape (data_points,), (data_points,input_dims), or [(data_points,)] * input_dims per channel with prediction X values. If a dictionary is passed, the index is the channel index or name.
            sigma (float): Number of standard deviations to display upwards and downwards.
            predict_y (boolean): Predict data values instead of function values.
            transformed (boolean): Return transformed data as used for training.

        Returns:
            numpy.ndarray: X prediction of shape (n,) for each channel.
            numpy.ndarray: Y mean prediction of shape (n,) for each channel.
            numpy.ndarray: Y lower prediction of uncertainty interval of shape (n,) for each channel.
            numpy.ndarray: Y upper prediction of uncertainty interval of shape (n,) for each channel.

        Examples:
            >>> model.predict(X)
        """
        if X is None:
            X = self.dataset.get_prediction_data()
        else:
            X = self.dataset._format_X(X)
        x = self._to_kernel_format(X)

        mu, var = self.gpr.predict(x, predict_y=predict_y, tensor=False)
        lower = mu - sigma*np.sqrt(var)
        upper = mu + sigma*np.sqrt(var)

        i = 0
        Mu = []
        Var = []
        Lower = []
        Upper = []
        for j in range(self.dataset.get_output_dims()):
            N = X[j].shape[0]
            Mu.append(np.squeeze(mu[i:i+N]))
            Var.append(np.squeeze(var[i:i+N]))
            Lower.append(np.squeeze(lower[i:i+N]))
            Upper.append(np.squeeze(upper[i:i+N]))
            i += N

        if not transformed:
            for j in range(self.dataset.get_output_dims()):
                Mu[j] = self.dataset[j].Y_transformer.backward(Mu[j], X[j])
                Lower[j] = self.dataset[j].Y_transformer.backward(Lower[j], X[j])
                Upper[j] = self.dataset[j].Y_transformer.backward(Upper[j], X[j])

        if len(self.dataset) == 1:
            return X[0], Mu[0], Lower[0], Upper[0]
        return X, Mu, Lower, Upper

    def K(self, X1, X2=None):
        """
        Evaluate the kernel at K(X1,X2).

        Args:
            X1 (list, dict): Array of shape (data_points,), (data_points,input_dims), or [(data_points,)] * input_dims per channel with prediction X values. If a dictionary is passed, the index is the channel index or name.
            X2 (list, dict): Same as X1 if None.

        Returns:
            numpy.ndarray: kernel evaluated at X1 and X2 of shape (n1,n2).

        Examples:
            >>> channel0 = np.array(['1987-05-20', '1987-05-21'])
            >>> channel1 = np.array([[2.5, 534.6], [3.5, 898.22], [4.5, 566.98]])
            >>> model.K([channel0,channel1])
        """
        X1 = self.dataset._format_X(X1)
        x1 = self._to_kernel_format(X1)
        if X2 is None:
            return self.gpr.K(x1)
        else:
            X2 = self.dataset._format_X(X2)
            x2 = self._to_kernel_format(X2)
            return self.gpr.K(x1, x2)

    def sample(self, X=None, n=None, predict_y=True, prior=False, transformed=False):
        """
        Sample n times from the kernel at input X .

        Args:
            X (list, dict): Array of shape (data_points,), (data_points,input_dims), or [(data_points,)] * input_dims per channel with prediction X values. If a dictionary is passed, the index is the channel index or name.
            n (int): Number of samples.
            predict_y (boolean): Predict data values instead of function values.
            prior (boolean): Sample from prior instead of posterior.
            transformed (boolean): Return transformed data as used for training.

        Returns:
            list: samples of shape len(X) for each channel if n is given.
            numpy.ndarray: sample of shape len(X) for each channel if n is None.

        Examples:
            >>> model.sample(n=10)
        """
        if X is None:
            X = self.dataset.get_prediction_data()
        else:
            X = self.dataset._format_X(X)
        x = self._to_kernel_format(X)
        samples = self.gpr.sample(Z=x, n=n, predict_y=predict_y)

        i = 0
        Samples = []
        for j in range(self.dataset.get_output_dims()):
            N = X[j].shape[0]
            if n is None:
                sample = np.squeeze(samples[i:i+N])
                if not transformed:
                    sample = self.dataset[j].Y_transformer.backward(sample, X[j])
                Samples.append(sample)
            else:
                sample = samples[i:i+N,:]
                for k in range(n):
                    if not transformed:
                        sample[:,k] = self.dataset[j].Y_transformer.backward(sample[:,k], X[j])
                Samples.append(sample)
            i += N
        if self.dataset.get_output_dims() == 1:
            return Samples[0]
        return Samples

    def plot_losses(self, title=None, figsize=(12,4), legend=True, errors=True, log=False):
        """
        Plot the losses and errors during training. In order to display the errors, make sure to set the error parameter when training.

        Args:
            title (str): Figure title.
            figsize (tuple): Figure size.
            legend (boolean): Show the legend.
            errors (boolean): Show the errors.
            log (boolean): Show in log scale.

        Returns:
            figure: Matplotlib figure.
            axis: Matplotlib axis.
        """
        if self.iters == 0:
            raise Exception("must be trained in order to plot the losses")

        fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
        x = np.arange(0,self.iters+1)
        ax.set_xlim(0, self.iters)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        if log:
            ax.set_yscale('log')

        ax.plot(x, self.losses, c='k', ls='-')

        legends = []
        legends.append(plt.Line2D([0], [0], ls='-', color='k', label='Loss'))
        if errors and x.shape[0] == self.errors.shape[0]:
            ax2 = ax.twinx()
            ax2.plot(x, self.errors, c='k', ls='-.')
            ax2.set_ylabel('Error')
            legends.append(plt.Line2D([0], [0], ls='-.', color='k', label='Error'))
            if log:
                ax2.set_yscale('log')

        if title is not None:
            fig.suptitle(title, fontsize=18)

        if legend:
            ax.legend(handles=legends)
        return fig, ax

    def plot_prediction(self, X=None, title=None, figsize=None, legend=True, errorbars=True, sigma=2, predict_y=True, transformed=False):
        """
        Plot the data including removed observations, latent function, and predictions of this model for each channel.

        Args:
            title (str): Set the title of the plot.
            figsize (tuple): Set the figure size.
            legend (boolean): Disable legend.
            errorbars (boolean): Plot data error bars if available.
            sigma (float): Number of standard deviations to display upwards and downwards.
            predict_y (boolean): Predict data values instead of function values.
            transformed (boolean): Display transformed Y data as used for training.

        Returns:
            matplotlib.figure.Figure: The figure.
            list of matplotlib.axes.Axes: List of axes.

        Examples:
            >>> fig, axes = dataset.plot(title='Title')
        """
        X, Mu, Lower, Upper = self.predict(X, sigma=sigma, predict_y=predict_y, transformed=transformed)
        if len(self.dataset) == 1:
            X = [X]
            Mu = [Mu]
            Lower = [Lower]
            Upper = [Upper]

        if figsize is None:
            figsize = (12,4*len(self.dataset))

        fig, ax = plt.subplots(len(self.dataset), 1, figsize=figsize, squeeze=False, constrained_layout=True)
        for j, data in enumerate(self.dataset):
            # TODO: ability to plot conditional or marginal distribution to reduce input dims
            if data.get_input_dims() > 2:
                raise ValueError("cannot plot more than two input dimensions")
            if data.get_input_dims() == 2:
                raise NotImplementedError("two dimensional input data not yet implemented") # TODO

            legends = []
            if errorbars and data.Y_err is not None:
                x, y = data.get_train_data(transformed=transformed)
                yl = data.Y[data.mask] - data.Y_err[data.mask]
                yu = data.Y[data.mask] + data.Y_err[data.mask]
                if transformed:
                    yl = data.Y_transformer.forward(yl, x)
                    yu = data.Y_transformer.forward(yu, x)
                x = x.astype(data.X_dtypes[0])
                ax[j,0].errorbar(x, y, [y-yl, yu-y], elinewidth=1.5, ecolor='lightgray', capsize=0, ls='', marker='')

            # prediction
            idx = np.argsort(X[j][:,0])
            x = X[j][idx,0].astype(data.X_dtypes[0])
            ax[j,0].plot(x, Mu[j][idx], ls=':', color='blue', lw=2)
            ax[j,0].fill_between(x, Lower[j][idx], Upper[j][idx], color='blue', alpha=0.3)
            label = 'Posterior Mean'
            legends.append(patches.Rectangle(
                (1, 1), 1, 1, fill=True, color='blue', alpha=0.3, lw=0, label='95% Error Bars'
            ))
            legends.append(plt.Line2D([0], [0], ls=':', color='blue', lw=2, label=label))

            xmin = min(np.min(data.X), np.min(X[j]))
            xmax = max(np.max(data.X), np.max(X[j]))
            if data.F is not None:
                if np.issubdtype(data.X.dtypes[0], np.datetime64):
                    dt = np.timedelta64(1,data.X.get_time_unit())
                    n = int((xmax-xmin) / dt) + 1
                    x = np.arange(xmin, xmax+np.timedelta64(1,'us'), dt, dtype=data.X.dtypes[0])
                else:
                    n = len(data.X)*10
                    x = np.linspace(xmin, xmax, n)

                y = data.F(x)
                if transformed:
                    y = data.Y_transformer.forward(y, x)

                ax[j,0].plot(x, y, 'r--', lw=1)
                legends.append(plt.Line2D([0], [0], ls='--', color='r', label='True'))

            if data.has_test_data():
                x, y = data.get_test_data(transformed=transformed)
                x = x.astype(data.X_dtypes[0])
                ax[j,0].plot(x, y, 'g.', ms=10)
                legends.append(plt.Line2D([0], [0], ls='', color='g', marker='.', ms=10, label='Latent'))

            x, y = data.get_train_data(transformed=transformed)
            x = x.astype(data.X_dtypes[0])
            ax[j,0].plot(x, y, 'r.', ms=10)
            legends.append(plt.Line2D([0], [0], ls='', color='r', marker='.', ms=10, label='Observations'))

            if 0 < len(data.removed_ranges[0]):
                for removed_range in data.removed_ranges[0]:
                    x0 = removed_range[0].astype(data.X_dtypes[0])
                    x1 = removed_range[1].astype(data.X_dtypes[0])
                    y0 = ax[j,0].get_ylim()[0]
                    y1 = ax[j,0].get_ylim()[1]
                    ax[j,0].add_patch(patches.Rectangle(
                        (x0, y0), x1-x0, y1-y0, fill=True, color='xkcd:strawberry', alpha=0.4, lw=0,
                    ))
                legends.insert(0, patches.Rectangle(
                    (1, 1), 1, 1, fill=True, color='xkcd:strawberry', alpha=0.4, lw=0, label='Removed Ranges'
                ))

            xmin = xmin.astype(data.X_dtypes[0])
            xmax = xmax.astype(data.X_dtypes[0])
            ax[j,0].set_xlim(xmin-(xmax-xmin)*0.001, xmax+(xmax-xmin)*0.001)
            ax[j,0].set_xlabel(data.X_labels[0])
            ax[j,0].set_ylabel(data.Y_label)
            ax[j,0].set_title(data.name if title is None else title, fontsize=14)

            if legend:
                ax[j,0].legend(handles=legends[::-1])
        return fig, ax

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
            start = [channel.X.min() for channel in self.dataset]
        if end is None:
            end = [channel.X.max() for channel in self.dataset]

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
            dist = [(channel.X.max()-channel.X.min())/4.0 for channel in self.dataset]

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

    def plot_correlation(self, title=None, figsize=(12,12)):
        """
        Plot the correlation matrix between each channel.

        Args:
            title (str): Figure title.
            figsize (tuple): Figure size.

        Returns:
            figure: Matplotlib figure.
            axis: Matplotlib axis.
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
        if title is not None:
            fig.suptitle(title, fontsize=18)

        output_dims = len(self.dataset)
        X = np.zeros((output_dims, 2))
        X[:,0] = np.arange(output_dims)
        K = self.gpr.K(X)

        # normalise
        diag_sqrt = np.sqrt(np.diag(K))
        K /= np.outer(diag_sqrt, diag_sqrt)

        im = ax.matshow(K, cmap='coolwarm', vmin=-1.0, vmax=1.0)
        for (i, j), z in np.ndenumerate(K):
            ax.text(j, i, '{:0.3f}'.format(z), ha='center', va='center', fontsize=14,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5, edgecolor='0.9'))

        ax.set_xticks(range(output_dims))
        ax.set_xticklabels(self.dataset.get_names(), fontsize=14)
        ax.set_yticks(range(output_dims))
        ax.set_yticklabels(self.dataset.get_names(), fontsize=14)
        ax.xaxis.set_ticks_position('top')
        return fig, ax

def _format_duration(s):
    if s < 60.0:
        return '%.3f seconds' % s

    s = math.floor(s)
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
    elif seconds == 1:
        duration += ' 1 second'
    return duration[1:]

def _format_time(s):
    return "%3d:%02d:%02d" % (int(s/3600), int((s%3600)/60), int(s%60))
