import os
import time
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
    Multioutput Gaussian proccess model. Can be either MOSM, CSM, SM-LMC or CONV.

    This class is used to use a multioutput GP model, train and test data can be added,
    the model can be used to train a predict.

    Example:

        ---TODO---

    Atributes:
        name ():
        data (obj, instance of mogptk.data):
        model ():
        Q (int): Number of components of the model.
        parameters ():
    """

    def __init__(self, name, data, Q):
        if not isinstance(data, list):
            data = [data]

        input_dims = data[0].get_input_dims()
        for channel in data:
            if channel.get_input_dims() != input_dims:
                raise Exception("all data channels must have the same amount of input dimensions (for now)")

        self.name = name
        self.data = [channel.copy() for channel in data]
        self.model = None
        self.Q = Q
        self.params = []
        self.fixed_params = []

    # overridden by specific models
    def _kernel(self):
        raise Exception("kernel not specified")
    
    # overridden by specific models
    def _transform_data(self):
        raise Exception("kernel not specified")

    # overridden by specific models
    def info(self):
        print("info() not implemented for kernel")

    # overridden by specific models
    def plot(self):
        print("plot() not implemented for kernel")

    def print(self):
        pd.set_option('display.max_colwidth', -1)
        df = pd.DataFrame(self.get_params())
        df.index.name = 'Q'
        display(df)

    def plot_data(self):
        for channel in self.data:
            channel.plot()

    def _update_params(self, trainables):
        for key, val in trainables.items():
            names = key.split("/")
            if len(names) == 5 and names[1] == 'kern' and names[2] == 'kernels':
                q = int(names[3])
                name = names[4]
                self.params[q][name] = val

    def get_input_dims(self):
        """
        Returns input dimension
        """
        # TODO: solve the different input dimension per channel case
        return self.data[0].get_input_dims()

    def get_output_dims(self):
        return len(self.data)

    def get_params(self):
        """
        Returns all parameters set for the kernel per component.
        """
        return self.params
        
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

    def set_pred(self, channel, x):
        """
        Sets prediction range

        TODO: Change so it can receive strings
        """
        self.data[channel].set_pred(x)

    def set_pred_range(self, channel, start=None, end=None, n=None, step=None):
        self.data[channel].set_pred_range(start, end, n, step)

    def fix_params(self, key):
        self.fixed_params.append(key)

    def unfix_params(self, key):
        self.fixed_params.remove(key)

    def save(self, filename):
        if self.model == None:
            raise Exception("build (and train) the model before doing predictions")

        if not filename.endswith(".mogptk"):
            filename += ".mogptk"

        try:
            os.remove(filename)
        except OSError:
            pass

        self.model.mogptk_type = self.__class__.__name__
        self.model.mogptk_name = self.name
        self.model.mogptk_data = []
        for channel in self.data:
            self.model.mogptk_data.append(channel._encode())
        self.model.mogptk_Q = self.Q
        self.model.mogptk_params = self.params
        self.model.mogptk_fixed_params = self.fixed_params

        with self.graph.as_default():
            with self.session.as_default():
                gpflow.Saver().save(filename, self.model)

    def build(self, likelihood=None, variational=False, sparse=False):
        """
        Build the model.

        Args:
            likelihood (gpflow.likelihoods): Likelihood to use from GPFlow, if None a default exact inference Gaussian likelihood is used.
            variational (bool): If True, use variational inference to approximate function values as Gaussian. If False it will use Monte carlo Markov Chain.
            sparse (bool): If True, will use sparse GP regression.
        """

        x, y = self._transform_data([channel.X[channel.mask] for channel in self.data], [channel.Y[channel.mask] for channel in self.data])

        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)
        with self.graph.as_default():
            with self.session.as_default():
                if likelihood == None:
                    if not sparse:
                        self.model = gpflow.models.GPR(x, y, self._kernel())
                    else:
                        # TODO: test if induction points are set
                        self.name += ' (sparse)'
                        self.model = gpflow.models.SGPR(x, y, self._kernel())
                elif not variational:
                    if not sparse:
                        self.name += ' (MCMC)'
                        self.model = gpflow.models.GPMC(x, y, self._kernel(), likelihood)
                    else:
                        self.name += ' (sparse MCMC)'
                        self.model = gpflow.models.SGPMC(x, y, self._kernel(), likelihood)
                else:
                    if not sparse:
                        self.name += ' (variational)'
                        self.model = gpflow.models.VGP(x, y, self._kernel(), likelihood)
                    else:
                        self.name += ' (sparse variational)'
                        self.model = gpflow.models.SVGP(x, y, self._kernel(), likelihood)
        
        for key in self.fixed_params:
            if hasattr(self.model.kern, 'kernels'):
                for kern in self.model.kern.kernels:
                    if hasattr(kern, key):
                        getattr(kern, key).trainable = False
                    else:
                        raise Exception("parameter name '%s' does not exist" % (key))
            else:
                if hasattr(self.model.kern, key):
                    getattr(self.model.kern, key).trainable = False
                else:
                    raise Exception("parameter name '%s' does not exist" % (key))

    def train(
        self,
        method='L-BFGS-B',
        likelihood=None,
        variational=False,
        sparse=False,
        plot=False,
        tol=1e-6,
        maxiter=2000,
        opt_params={},
        params={},
        export_graph=False):
        """
        Builds and trains the model using the kernel and its parameters.

        For different optimizers, see scipy.optimize.minimize.
        It can be bounded by a maximum number of iterations, disp will output final
        optimization information. When using the 'Adam' optimizer, a
        learning_rate can be set.

        Args:
            method (str): Optimizer to use, if "Adam" is chosen,
                gpflow.training.Adamoptimizer will be used, otherwise the passed scipy
                optimizer is used. Default to scipy 'L-BFGS-B'.
            likelihood (gpflow.likelihoods): Likelihood to use from GPFlow, if None a default exact inference Gaussian likelihood is used.
            variational (bool): If True, use variational inference to approximate function values as Gaussian. If False it will use Monte carlo Markov Chain.
            sparse (bool): If True, will use sparse GP regression.
            plot (bool): If true will plot the spectrum. Default to False.
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
        
        self.build(likelihood, variational, sparse)

        with self.graph.as_default():
            with self.session.as_default():
                if export_graph:
                    def get_tensor(name):
                        return self.graph.get_tensor_by_name('GPR-' + self.model._index + '/likelihood_1/' + name + ':0')

                    #print([n.name for n in tf.get_default_graph().as_graph_def().node])

                    writer = tf.summary.FileWriter("log", self.graph)
                    K_summary = tf.summary.histogram('K', get_tensor('K'))

                step_i = 0
                def step(theta):
                    nonlocal step_i
                    if export_graph:
                        #writer.add_summary(self.session.run(likelihood_summary), step_i)
                        #writer.add_summary(self.session.run(prior_summary), step_i)
                        writer.add_summary(self.session.run(K_summary), step_i)
                    step_i += 1

                if method == "Adam":
                    opt = gpflow.training.AdamOptimizer(**opt_params)
                    opt.minimize(self.model, anchor=True, **params)
                else:
                    opt = gpflow.train.ScipyOptimizer(method=method, tol=tol, **opt_params)
                    opt.minimize(self.model, anchor=True, step_callback=step, maxiter=maxiter, **params)

                self._update_params(self.model.read_trainables())

        print("Done in ", (time.time() - start_time)/60, " minutes")

        if plot:
            self.plot()

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
        if self.model == None:
            raise Exception("build (and train) the model before doing predictions")

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

