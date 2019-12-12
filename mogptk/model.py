import os
import time
import dill
import numpy as np
from pprint import pprint
import gpflow
import tensorflow as tf
import pandas as pd
from .data import _detransform
from .dataset import DataSet
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import logging
logging.getLogger('tensorflow').propagate = False

def LoadModel(filename):
    """
    Load a model from a given file that was previously saved (see the `model.save()` function).
    """
    if not filename.endswith(".mogptk"):
        filename += ".mogptk"

    graph = tf.Graph()
    session = tf.Session(graph=graph)
    with graph.as_default():
        with session.as_default():
            model = gpflow.saver.Saver().load(filename)
            model_type = model.mogptk_type
            name = model.mogptk_name
            data = dill.loads(eval(model.mogptk_data))
            Q = model.mogptk_Q
            params = model.mogptk_params

    if model_type == 'SM':
        m = SM(data, Q, name)
    elif model_type == 'MOSM':
        m = MOSM(data, Q, name)
    elif model_type == 'CG':
        m = CG(data, Q, name)
    elif model_type == 'CSM':
        m = CSM(data, Q, name)
    elif model_type == 'SM_LMC':
        m = SM_LMC(data, Q, name)
    else:
        raise Exception("unknown model type '%s'" % (model_type))

    m.model = model
    m.params = params
    m.graph = graph
    m.session = session
    m.build() # TODO: should not be necessary
    return m

class model:
    def __init__(self, name, dataset):
        """
        Base class for Multi-Output Gaussian process models. See subclasses for instantiation.
            
        Args:
            name (string): Name of the model.
            dataset (DataSet): DataSet with Data objects for all the channels.
        """
        
        if not isinstance(dataset, DataSet):
            dataset = DataSet(dataset)
        if dataset.get_output_dims() == 0:
            raise Exception("dataset must have at least one channel")
        if len(set(dataset.get_names())) != len(dataset.get_names()):
            raise Exception("all data channels must have unique names")
        if len(set(dataset.get_input_dims())) != 1:
            raise Exception("all data channels must have the same amount of input dimensions")

        self.name = name
        self.dataset = dataset
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)
    
    def _build(self, kernel, likelihood, variational, sparse, like_params):
        """
        Args:
            kernel (gpflow.Kernel): Kernel to use.
            likelihood (gpflow.Likelihood): Likelihood to use from GPFlow, if None a default exact inference Gaussian likelihood is used.
            variational (bool): If True, use variational inference to approximate function values as Gaussian. If False it will use Monte Carlo Markov Chain.
            sparse (bool): If True, will use sparse GP regression.
            like_params (dict): Parameters to GPflow likelihood.
        """

        x, y = self.dataset.to_kernel()
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

    def get_params(self):
        """
        Returns all parameters set for the kernel per component.
        """
        params = []
        if hasattr(self.model.kern, 'kernels'):
            for kernel_i, kernel in enumerate(self.model.kern.kernels):
                params.append({})
                for param_name, param_val in kernel.__dict__.items():
                    if isinstance(param_val, gpflow.params.parameter.Parameter):
                        params[kernel_i][param_name] = param_val.read_value()
        else:
            params.append({})
            for param_name, param_val in self.model.kern.__dict__.items():
                if isinstance(param_val, gpflow.params.parameter.Parameter):
                    params[0][param_name] = param_val.read_value()
        return params

    def set_param(self, q, key, val):
        """
        Sets an initial kernel parameter prior to optimizations for component 'q'
        with key the parameter name.

        Args:
            q (int): Component of kernel.
            key (str): Name of component.
            val (float, ndarray): Value of parameter.
        """
        if isinstance(val, (int, float)):
            val = np.array(val)
        if not isinstance(val, np.ndarray):
            raise Exception("value %s of type %s is not a number type or ndarray" % (val, type(val)))

        if hasattr(self.model.kern, 'kernels'):
            if q < 0 or len(self.model.kern.kernels) <= q:
                raise Exception("qth component %d does not exist" % (q))
            kern = self.model.kern.kernels[q].__dict__
        else:
            if q != 0:
                raise Exception("qth component %d does not exist" % (q))
            kern = self.model.kern.__dict__

        if key not in kern or not isinstance(kern[key], gpflow.params.parameter.Parameter):
            raise Exception("parameter name '%s' does not exist" % (key))

        if kern[key].shape != val.shape:
            raise Exception("parameter name '%s' must have shape %s and not %s" % (key, kern[key].shape, val.shape))

        with self.graph.as_default():
            with self.session.as_default():
                kern[key].assign(val)

    def fix_param(self, key):
        """
        Make parameter untrainable (undo with `unfix_param`).

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

    def unfix_param(self, key):
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

        print("Done in ", (time.time() - start_time)/60, " minutes")

    ################################################################################
    # Predictions ##################################################################
    ################################################################################

    def predict(self, pred_x=None, plot=False):
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
        if pred_x is not None:
            self.dataset.set_pred(self.name, pred_x)

        # predict with model
        x = self.dataset.to_kernel_pred()
        if len(x) == 0:
                raise Exception('no prediction x range set, use pred_x argument or set manually using DataSet.set_pred() or Data.set_pred()')
        with self.graph.as_default():
            with self.session.as_default():
                mu, var = self.model.predict_f(x)
        self.dataset.from_kernel_pred(self.name, mu, var)
        
        if plot:
            self.plot_prediction()

        #return self.dataset.get_pred(self.name)

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
        x_train, y_train = self.dataset.get_data()
        x_all, y_all = self.dataset.get_all()
        x_pred, mu, lower, upper = self.dataset.get_pred(self.name)

        n_dim = self.dataset.get_output_dims()
        if n_dim == 1:
            grid = (1, 1)
        elif grid is None:
            grid = (int(np.ceil(n_dim/2)), 2)

        if (grid[0] * grid[1]) < n_dim:
            raise Exception('grid not big enough for all channels')

        fig, axes = plt.subplots(grid[0], grid[1], sharex=False, figsize=figsize)
        axes = np.array(axes).reshape(-1)

        colors = list(matplotlib.colors.TABLEAU_COLORS)
        for i in range(n_dim):
            axes[i].plot(x_train[i, :, 0], y_train[i], '.k', label='Train', ms=5)
            axes[i].plot(x_all[i, :, 0], y_all[i], '--', label='Test', c='gray',lw=1.4, zorder=5)
            
            axes[i].plot(x_pred[i, :, 0], mu[i], label='Post.Mean', c=colors[i], zorder=1)
            axes[i].fill_between(x_pred[i, :, 0].reshape(-1),
                lower[i],
                upper[i],
                label='95% c.i',
                color=colors[i],
                alpha=0.4)
            
            # axarr[i].legend(ncol=4, loc='upper center', fontsize=8)

            # axarr[i].locator_params(tight=True, nbins=6)
            axes[i].xaxis.set_major_locator(plt.MaxNLocator(6))

            formatter = matplotlib.ticker.FuncFormatter(lambda x,pos: self.dataset.get(i).formatters[0]._format(x))
            axes[i].xaxis.set_major_formatter(formatter)

            # set channels name
            if names is not None:
                axes[i].set_title(names[i])
            else:
                channel_name = self.dataset.get_names()[i]
                if channel_name != '':
                    axes[i].set_title(channel_name)
                elif n_dim == 1:
                    pass
                else:
                    axes[i].set_title('Channel ' + str(i))

            # set y lims
            if ylims is not None:
                axes[i].set_ylim(ylims[i]) 
            
        plt.suptitle(title, y=1.02)
        plt.tight_layout()

        data_dict = {
            'x_train':x_train,
            'y_train':y_train,
            'x_all':x_all,
            'y_all':y_all,
            'y_pred':mu,
            'low_ci':lower,
            'hi_ci':upper,
        }

        if ret_fig:
            return fig, axes, data_dict

