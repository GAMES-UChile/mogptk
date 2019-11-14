import os
import time
import dill
import numpy as np
from pprint import pprint
import gpflow
import tensorflow as tf
import pandas as pd
from .data import _detransform

import logging
logging.getLogger('tensorflow').propagate = False

class model:
    """
    Base class for Multi-Output Gaussian process models. See subclasses for instantiation.
        
    Args:
        data (Data,list of Data): Data object or list of Data objects for each channel.
        name (string): Name of the model.
        kernel (gpflow.Kernel): Kernel to use.
        likelihood (gpflow.Likelihood): Likelihood to use from GPFlow, if None a default exact inference Gaussian likelihood is used.
        variational (bool): If True, use variational inference to approximate function values as Gaussian. If False it will use Monte Carlo Markov Chain.
        sparse (bool): If True, will use sparse GP regression.
        like_params (dict): Parameters to GPflow likelihood.
    """

    def __init__(self, data, name, kernel, likelihood, variational, sparse, like_params):
        if not isinstance(data, list):
            data = [data]

        names = set()
        input_dims = data[0].get_input_dims()
        for channel in data:
            if channel.get_input_dims() != input_dims:
                raise Exception("all data channels must have the same amount of input dimensions")
            if channel.name != "":
                if channel.name in names:
                    raise Exception("data channels must have different names")
                names.add(channel.name)

        self.name = name
        self.data = [channel.copy() for channel in data]
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)

        x, y = self._transform_data([channel.X[channel.mask] for channel in self.data], [channel.Y[channel.mask] for channel in self.data])
        with self.graph.as_default():
            with self.session.as_default():
                # Gaussian likelihood
                if likelihood == None:
                    if not sparse:
                        self.model = gpflow.models.GPR(x, y, kernel)
                    else:
                        # TODO: test if induction points are set
                        self.name += ' (sparse)'
                        self.model = gpflow.models.SGPR(x, y, kernel)
                # MCMC
                elif not variational:
                    self.likelihood = likelihood(**like_params)
                    if not sparse:
                        self.name += ' (MCMC)'
                        self.model = gpflow.models.GPMC(x, y, kernel, self.likelihood)
                    else:
                        self.name += ' (sparse MCMC)'
                        self.model = gpflow.models.SGPMC(x, y, kernel, self.likelihood)
                # Variational
                else:
                    self.likelihood = likelihood(**like_params)
                    if not sparse:
                        self.name += ' (variational)'
                        self.model = gpflow.models.VGP(x, y, kernel, self.likelihood)
                    else:
                        self.name += ' (sparse variational)'
                        self.model = gpflow.models.SVGP(x, y, kernel, self.likelihood)

    # overridden by specific models
    def _kernel(self):
        raise Exception("kernel not specified")
    
    # overridden by specific models
    def _transform_data(self):
        raise Exception("kernel not specified")

    # overridden by specific models
    def info(self):
        """
        Information about the model.
        """
        print("info() not implemented for kernel")

    # overridden by specific models
    def plot(self):
        """
        Plot the model.
        """
        print("plot() not implemented for kernel")

    ################################################################

    def print_params(self):
        """
        Print the parameters of the model in a table.
        """
        pd.set_option('display.max_colwidth', -1)
        df = pd.DataFrame(self.get_params())
        df.index.name = 'Q'
        display(df)

    def _update_params(self, trainables):
        for key, val in trainables.items():
            names = key.split("/")
            if len(names) == 5 and names[1] == 'kern' and names[2] == 'kernels':
                q = int(names[3])
                name = names[4]
                self.params[q][name] = val

    def get_input_dims(self):
        """
        Returns the number of input dimensions of the data.
        """
        return self.data[0].get_input_dims()  # all channels have the same number of input dimensions

    def get_output_dims(self):
        """
        Returns the number of output dimensions of the data, i.e. the number of channels.
        """
        return len(self.data)

    def get_channel(self, channel):
        """
        Return Data for a channel.

        Args:
            channel (int,string): Index or name of the channel.
        """
        if isinstance(channel, int):
            if channel < len(self.data):
                return self.data[channel]
        elif isinstance(channel, str):
            for data in self.data:
                if data.name == channel:
                    return data
        raise ValueError("channel '%d' does not exist" % (channel))

    def get_params(self):
        """
        Returns all parameters set for the kernel per component.
        """
        return self.params

    def get_x_pred(self):
        """
        Returns the input used in the last prediction
        """
        return [channel.X_pred for channel in self.data]

        
    def _get_param_across(self, name='mixture_means'):
        """
        Get all the name parameters across all components.
        """
        return np.array([self.params[q][name] for q in range(self.Q)])

    def set_param(self, q, key, val):
        """
        Sets an initial kernel parameter prior to optimizations for component 'q'
        with key the parameter name.

        Args:
            q (int): Component of kernel.
            key (str): Name of component.
            val (float, ndarray): Value of parameter.
        """
        if q < 0 or len(self.params) <= q:
            raise Exception("qth component %d does not exist" % (q))
        if key not in self.params[q]:
            raise Exception("parameter name '%s' does not exist" % (key))

        self.params[q][key] = val

    def fix_param(self, key):
        """
        Make parameter untrainable (undo with `train_param`).

        Args:
            key (string): Name of the parameter.
        """
        if hasattr(self.model.kern, 'kernels'):
            for kernel_i, kernel in enumerate(self.model.kern.kernels):
                for param_name, param_val in kernel.__dict__.items():
                    if param_name == key and isinstance(param_val, gpflow.params.parameter.Parameter):
                        getattr(self.model.kern.kernels[kernel_i], param_name).trainable = False
        else:
            for param_name, param_val in self.model.kern.__dict__.items():
                if param_name == key and isinstance(param_val, gpflow.params.parameter.Parameter):
                    getattr(self.model.kern, param_name).trainable = False

    def train_param(self, key):
        """
        Make parameter trainable (that was previously fixed, see `fix_param`).

        Args:
            key (string): Name of the parameter.
        """
        if hasattr(self.model.kern, 'kernels'):
            for kernel_i, kernel in enumerate(self.model.kern.kernels):
                for param_name, param_val in kernel.__dict__.items():
                    if param_name == key and isinstance(param_val, gpflow.params.parameter.Parameter):
                        getattr(self.model.kern.kernels[kernel_i], param_name).trainable = True
        else:
            for param_name, param_val in self.model.kern.__dict__.items():
                if param_name == key and isinstance(param_val, gpflow.params.parameter.Parameter):
                    getattr(self.model.kern, param_name).trainable = True

    def save(self, filename):
        if not filename.endswith(".mogptk"):
            filename += ".mogptk"

        try:
            os.remove(filename)
        except OSError:
            pass

        self.model.mogptk_type = self.__class__.__name__
        self.model.mogptk_name = self.name
        self.model.mogptk_data = str(dill.dumps(self.data))
        self.model.mogptk_Q = self.Q
        self.model.mogptk_params = self.params

        with self.graph.as_default():
            with self.session.as_default():
                gpflow.Saver().save(filename, self.model)

    def train(
        self,
        method='L-BFGS-B',
        tol=1e-6,
        maxiter=2000,
        opt_params={},
        params={},
        export_graph=False):
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
            maxiter (int): Maximum number of iterations. Defaults to 2000.
            opt_params (dict): Aditional dictionary with parameters on optimizer.
                If method is 'Adam' see:
                https://github.com/GPflow/GPflow/blob/develop/gpflow/training/tensorflow_optimizer.py
                If method is in scipy-optimizer see:
                https://github.com/GPflow/GPflow/blob/develop/gpflow/training/scipy_optimizer.py
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
            params (dict): Aditional dictionary with parameters to minimice. 
                See https://github.com/GPflow/GPflow/blob/develop/gpflow/training/optimizer.py
                for more details.
            export_graph (bool): Default to False.
        """
        start_time = time.time()
        with self.graph.as_default():
            with self.session.as_default():
                if export_graph:
                    def get_tensor(name):
                        return self.graph.get_tensor_by_name('GPR-' + self.model._index + '/likelihood_1/' + name + ':0')
                    writer = tf.summary.FileWriter("log", self.graph)
                    K_summary = tf.summary.histogram('K', get_tensor('K'))

                step_i = 0
                def step(theta):
                    nonlocal step_i
                    if export_graph:
                        writer.add_summary(self.session.run(K_summary), step_i)
                    step_i += 1

                if method == "Adam":
                    opt = gpflow.training.AdamOptimizer(**opt_params)
                    opt.minimize(self.model, anchor=True, **params)
                else:
                    opt = gpflow.train.ScipyOptimizer(method=method, tol=tol, **opt_params)
                    opt.minimize(self.model, anchor=True, step_callback=step, maxiter=maxiter, disp=True, **params)

                self._update_params(self.model.read_trainables())

        print("Done in ", (time.time() - start_time)/60, " minutes")

    ################################################################################
    # Predictions ##################################################################
    ################################################################################

    def predict(self, x_pred=None):
        """
        Predict with model.

        Will make a prediction using x as input. If no input value is passed, the prediction will 
        be made with atribute self.X_pred that can be setted with other functions.
        It returns the X, Y_mu, Y_var values per channel.

        Args:
            x_pred (list, dict): Dictionary where keys are channel index and elements numpy arrays with channel inputs.

        Returns:
            Y_mu_pred, Y_lower_ci_predicted, Y_upper_ci_predicted: 
            Prediction output and confidence interval of 95% of the model (Upper and lower bounds).

        """
        # if user pass a prediction input
        if x_pred is not None:
            for i, x_channel in enumerate(x_pred):
                self.data[i].set_pred(x_channel)

        # check if there is some prediction seted
        for channel in self.data:
            if channel.X_pred.size == 0:
                raise Exception('no prediction value set, use x_pred argument or set manually using data.set_pred().')

        x = self._transform_data([channel.X_pred for channel in self.data])

        # predict with model
        with self.graph.as_default():
            with self.session.as_default():
                mu, var = self.model.predict_f(x)

        # reshape for channels
        i = 0
        for channel in self.data:
            n = channel.X_pred.shape[0]
            if n != 0:
                channel.Y_mu_pred = mu[i:i+n].reshape(1, -1)[0]
                channel.Y_var_pred = var[i:i+n].reshape(1, -1)[0]
                i += n

        # inverse transformations
        Y_mu_predicted = []
        Y_upper_ci_predicted = []
        Y_lower_ci_predicted = []

        for channel in self.data:
            # detransform mean
            y_pred_detrans = _detransform(channel.transformations, channel.X_pred, channel.Y_mu_pred)
            Y_mu_predicted.append(y_pred_detrans)
            
            # upper confidence interval
            u_ci = channel.Y_mu_pred + 2 * np.sqrt(channel.Y_var_pred)
            u_ci = _detransform(channel.transformations, channel.X_pred, u_ci)
            Y_upper_ci_predicted.append(u_ci)

            # lower confidence interval
            l_ci = channel.Y_mu_pred - 2 * np.sqrt(channel.Y_var_pred)
            l_ci = _detransform(channel.transformations, channel.X_pred, l_ci)
            Y_lower_ci_predicted.append(l_ci)

        return Y_mu_predicted, Y_lower_ci_predicted, Y_upper_ci_predicted

