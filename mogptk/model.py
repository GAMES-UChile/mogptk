import os
import time
import dill
import numpy as np
from pprint import pprint
import gpflow
import tensorflow as tf
import pandas as pd
from .data import _detransform
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import logging
logging.getLogger('tensorflow').propagate = False

class model:
    """
    Base class for Multi-Output Gaussian process models. See subclasses for instantiation.
    """

    def __init__(self, name, data, Q):
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
        self.model = None
        self.Q = Q
        self.params = [] # TODO: remove and use GPflow?

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
        if self.model == None:
            raise Exception("build the model before disabling training on parameter")

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
        if self.model == None:
            raise Exception("build the model before enabling training on parameter")

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
        if self.model == None:
            raise Exception("build the model before saving")

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

    def build(self, likelihood=None, variational=False, sparse=False, like_params={}):
        """
        Build the model.

        Args:
            likelihood (gpflow.likelihoods): Likelihood to use from GPFlow, if None
                a default exact inference Gaussian likelihood is used.
            variational (bool): If True, use variational inference to approximate
                function values as Gaussian. If False it will use Monte carlo Markov Chain.
            sparse (bool): If True, will use sparse GP regression.
        """

        x, y = self._transform_data([channel.X[channel.mask] for channel in self.data], [channel.Y[channel.mask] for channel in self.data])

        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)
        with self.graph.as_default():
            with self.session.as_default():
                # Gaussian likelihood
                if likelihood == None:
                    if not sparse:
                        self.model = gpflow.models.GPR(x, y, self._kernel())
                    else:
                        # TODO: test if induction points are set
                        self.name += ' (sparse)'
                        self.model = gpflow.models.SGPR(x, y, self._kernel())
                # MCMC
                elif not variational:
                    self.likelihood = likelihood(**like_params)
                    if not sparse:
                        self.name += ' (MCMC)'
                        self.model = gpflow.models.GPMC(x, y, self._kernel(), self.likelihood)
                    else:
                        self.name += ' (sparse MCMC)'
                        self.model = gpflow.models.SGPMC(x, y, self._kernel(), self.likelihood)
                # Variational
                else:
                    self.likelihood = likelihood(**like_params)
                    if not sparse:
                        self.name += ' (variational)'
                        self.model = gpflow.models.VGP(x, y, self._kernel(), self.likelihood)
                    else:
                        self.name += ' (sparse variational)'
                        self.model = gpflow.models.SVGP(x, y, self._kernel(), self.likelihood)

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
        if self.model == None:
            raise Exception("build the model before training")

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
        if self.model == None:
            raise Exception("build the model before doing predictions")

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



    def plot_prediction(self, grid=None, figsize=(12, 8), ylims=None, names=None, title='', ret_fig=False):

        """
        Plot training points, all data and prediction for training range for all channels.

        Args:
            Model (mogptk.Model object): Model to use.
            grid (tuple) : Tuple with the 2 dimensions of the grid.
            figsize(tuple): Figure size, default to (12, 8).
            ylims(list): List of tuples with limits for Y axis for
                each channel.
            Names(list): List of the names of each title.
            title(str): Title of the plot.
            ret_fig(bool): If true returns the matplotlib figure, 
                array of axis and dictionary with all the points used.

        TODO: Add case for single output SM kernel.
        """
        # get data
        x_train = [c.X[c.mask] for c in self.data]
        y_train = [_detransform(c.transformations, c.X[c.mask], c.Y[c.mask]) for c in self.data]
        x_all = [c.X for c in self.data]
        y_all = [_detransform(c.transformations, c.X, c.Y) for c in self.data]
        x_pred = [c.X for c in self.data]

        n_dim = self.get_output_dims()

        if n_dim == 1:
            grid = (1, 1)
        elif grid is None:
            grid = (int(np.ceil(n_dim/2)), 2)

        if (grid[0] * grid[1]) < n_dim:
            raise Exception('Grid not big enough for all channels')

        # predict with model
        mean_pred, lower_ci, upper_ci = self.predict(x_pred)

        # create plot
        f, axarr = plt.subplots(grid[0], grid[1], sharex=False, figsize=figsize)

        if not isinstance(axarr, np.ndarray):
            axarr = np.array([axarr])

        axarr = axarr.reshape(-1)

        color_palette = mcolors.TABLEAU_COLORS
        color_names = list(mcolors.TABLEAU_COLORS)

        # plot
        for i in range(n_dim):
            color = color_palette[color_names[i]]

            axarr[i].plot(x_train[i][:, 0], y_train[i], '.k', label='Train', ms=5)
            axarr[i].plot(x_all[i][:, 0], y_all[i], '--', label='Test', c='gray',lw=1.4, zorder=5)
            
            axarr[i].plot(x_pred[i][:, 0], mean_pred[i], label='Post.Mean', c=color, zorder=1)
            axarr[i].fill_between(x_pred[i][:, 0].reshape(-1),
                                  lower_ci[i],
                                  upper_ci[i],
                                  label='95% c.i',
                                  color=color,
                                  alpha=0.4)
            
            # axarr[i].legend(ncol=4, loc='upper center', fontsize=8)

            # axarr[i].locator_params(tight=True, nbins=6)
            axarr[i].xaxis.set_major_locator(plt.MaxNLocator(6))

            formatter = matplotlib.ticker.FuncFormatter(lambda x,pos: self.data[i].formatters[0]._format(x))
            axarr[i].xaxis.set_major_formatter(formatter)

            # set channels name
            if names is not None:
                axarr[i].set_title(names[i])
            else:
                channel_name = self.data[i].name
                if channel_name != '':
                    axarr[i].set_title(channel_name)
                elif n_dim == 1:
                    pass
                else:
                    axarr[i].set_title('Channel ' + str(i))

            # set y lims
            if ylims is not None:
                axarr[i].set_ylim(ylims[i]) 
            
        plt.suptitle(title, y=1.02)
        plt.tight_layout()

        data_dict = {
        'x_train':x_train,
        'y_train':y_train,
        'x_all':x_all,
        'y_all':y_all,
        'y_pred':mean_pred,
        'low_ci':lower_ci,
        'hi_ci':upper_ci,
        }

        if ret_fig:
            return f, axarr, data_dict

