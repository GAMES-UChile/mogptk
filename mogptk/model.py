import os
import json
import time
import numpy as np
import gpflow
import tensorflow as tf
from .dataset import DataSet
import matplotlib
import matplotlib.pyplot as plt

from IPython.display import display, HTML
from tabulate import tabulate

import logging
logging.getLogger('tensorflow').propagate = False
logging.getLogger('tensorflow').setLevel(logging.ERROR)

gpflow.config.set_default_positive_minimum(1e-6)
#gpflow.config.set_default_positive_bijector("exp")

class model:
    def __init__(self, name, dataset):
        #"""
        #Base class for Multi-Output Gaussian process models. See subclasses for instantiation.
        #    
        #Args:
        #    name (str): Name of the model.
        #    dataset (mogptk.dataset.DataSet): DataSet with Data objects for all the channels.
        #    When a (list or dict of) Data object is passed, it will automatically be converted to a DataSet.
        #"""
        
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
    
    def _build(self, kernel, likelihood, variational, sparse, like_params):
        """
        Build the model using the given kernel and likelihood. The variational and sparse booleans decide which GPflow model will be used.

        Args:
            kernel (gpflow.Kernel): Kernel to use.
            likelihood (gpflow.likelihoods): Likelihood to use from GPFlow, if None
                a default exact inference Gaussian likelihood is used.
            variational (bool): If True, use variational inference to approximate
                function values as Gaussian. If False it will use Monte carlo Markov Chain.
            sparse (bool): If True, will use sparse GP regression.
            like_params (dict): Parameters to GPflow likelihood.
        """

        x, y = self.dataset.to_kernel()
        # Gaussian likelihood
        if likelihood == None:
            if not sparse:
                self.model = gpflow.models.GPR((x, y), kernel)
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

    ################################################################

    def print_params(self):
        """
        Print the parameters of the model in a table.

        Examples:
            >>> model.print_params()
        """
        with np.printoptions(precision=3, floatmode='fixed'):
            try:
                get_ipython
                table = ''
                for q, params in enumerate(self.get_params()):
                    contents = []
                    output_dims = self.dataset.get_output_dims()
                    for j in range(output_dims):
                        content = '<td>%s</td>' % (self.dataset[j].get_name(),)

                        values = []
                        for key in params.keys():
                            val = params[key]
                            if val.ndim == 1 and val.shape[0] == output_dims:
                                values.append('%.3f' % (val[j],))
                            elif val.ndim == 2 and val.shape[1] == output_dims:
                                values.append(str(val[:,j]))
                            else:
                                values.append(str(val))
                        content += '<td>' + '</td><td>'.join(values) + '</td>'
                        contents.append(content)

                    header = '<tr><th>Q=%d</th><th>' % (q,) + '</th><th>'.join(params.keys()) + '</th></tr>'
                    contents = '<tr>' + '</tr><tr>'.join(contents) + '</tr>'
                    table += header + contents
                display(HTML('<table>%s</table>' % (table)))

                likelihood = self.get_likelihood_params()
                table = '<tr><th>Likelihood</th>'
                for key in likelihood:
                    table += '<th>%s</th>' % (key,)
                table += '</tr><tr><td></td>'
                for key in likelihood:
                    table += '<td>%s</td>' % (likelihood[key],)
                table += '</tr>'
                display(HTML('<table>%s</table>' % (table)))
            except:
                table = ''
                for q, params in enumerate(self.get_params()):
                    contents = []
                    output_dims = self.dataset.get_output_dims()
                    for j in range(output_dims):
                        content = [self.dataset[j].get_name()]
                        for key in params.keys():
                            val = params[key]
                            if val.ndim == 1 and val.shape[0] == output_dims:
                                content.append('%.3f' % (val[j],))
                            elif val.ndim == 2 and val.shape[1] == output_dims:
                                content.append(str(val[:,j]))
                            else:
                                content.append(str(val))
                        contents.append(content)

                    headers = ["Q=%d" % (q,)]
                    for key in params.keys():
                        headers.append(key)
                    print(tabulate(contents, headers=headers))
                    print()

                likelihood = self.get_likelihood_params()
                content = ['']
                for key in likelihood.keys():
                    val = likelihood[key]
                    if val.ndim == 1 and val.shape[0] == output_dims:
                        content.append('%.3f' % (val[j],))
                    elif val.ndim == 2 and val.shape[1] == output_dims:
                        content.append(str(val[:,j]))
                    else:
                        content.append(str(val))

                headers = ["Likelihood"]
                for key in likelihood.keys():
                    headers.append(key)
                print(tabulate([content], headers=headers))

    def get_params(self):
        """
        Returns all parameters set for the kernel per component.

        Examples:
            >>> params = model.get_params()
        """

        params = []
        if hasattr(self.model.kernel, 'kernels'):
            for kernel_i, kernel in enumerate(self.model.kernel.kernels):
                params.append({})
                for param_name, param_val in kernel.__dict__.items():
                    if isinstance(param_val, gpflow.base.Parameter):
                        params[kernel_i][param_name] = param_val.read_value().numpy()
        else:
            params.append({})
            for param_name, param_val in self.model.kernel.__dict__.items():
                if isinstance(param_val, gpflow.base.Parameter):
                    params[0][param_name] = param_val.read_value().numpy()
        return params

    def get_likelihood_params(self):
        """
        Returns all parameters set for the likelihood.

        Examples:
            >>> params = model.get_likelihood_params()
        """
        params = {}
        for param_name, param_val in self.model.likelihood.__dict__.items():
            if isinstance(param_val, gpflow.base.Parameter):
                params[param_name] = param_val.read_value().numpy()
        return params

    def get_param(self, q, key):
        """
        Gets a kernel parameter for component 'q' with key the parameter name.

        Args:
            q (int): Component of kernel.
            key (str): Name of component.
            
        Returns:
            val (numpy.ndarray): Value of parameter.

        Examples:
            >>> val = model.get_param(0, 'variance') # for Q=0 get the parameter called 'variance'
        """
        if hasattr(self.model.kernel, 'kernels'):
            if q < 0 or len(self.model.kernel.kernels) <= q:
                raise Exception("qth component %d does not exist" % (q))
            kern = self.model.kernel.kernels[q].__dict__
        else:
            if q != 0:
                raise Exception("qth component %d does not exist" % (q))
            kern = self.model.kernel.__dict__
        
        if key not in kern or not isinstance(kern[key], gpflow.base.Parameter):
            raise Exception("parameter name '%s' does not exist" % (key))
    
        return kern[key].read_value().numpy()

    def set_param(self, q, key, val):
        """
        Sets a kernel parameter for component 'q' with key the parameter name.

        Args:
            q (int): Component of kernel.
            key (str): Name of component.
            val (float, numpy.ndarray): Value of parameter.

        Examples:
            >>> model.set_param(0, 'variance', np.array([5.0, 3.0])) # for Q=0 set the parameter called 'variance'
        """
        if isinstance(val, (int, float, list)):
            val = np.array(val)
        if not isinstance(val, np.ndarray):
            raise Exception("value %s of type %s is not a number type or ndarray" % (val, type(val)))

        if hasattr(self.model.kernel, 'kernels'):
            if q < 0 or len(self.model.kernel.kernels) <= q:
                raise Exception("qth component %d does not exist" % (q))
            kern = self.model.kernel.kernels[q].__dict__
        else:
            if q != 0:
                raise Exception("qth component %d does not exist" % (q))
            kern = self.model.kernel.__dict__

        if key not in kern or not isinstance(kern[key], gpflow.base.Parameter):
            raise Exception("parameter name '%s' does not exist" % (key))

        if kern[key].shape != val.shape:
            raise Exception("parameter name '%s' must have shape %s and not %s" % (key, kern[key].shape, val.shape))

        for i, v in np.ndenumerate(val):
            if v < gpflow.config.default_positive_minimum():
                val[i] = gpflow.config.default_positive_minimum()

        kern[key].assign(val)

    def set_likelihood_param(self, key, val):
        """
        Sets a likelihood parameter with key the parameter name.

        Args:
            key (str): Name of component.
            val (float, ndarray): Value of parameter.

        Examples:
            >>> model.set_likelihood_param('variance', np.array([5.0, 3.0])) # set the parameter called 'variance'
        """
        if isinstance(val, (int, float, list)):
            val = np.array(val)
        if not isinstance(val, np.ndarray):
            raise Exception("value %s of type %s is not a number type or ndarray" % (val, type(val)))

        likelihood = self.model.likelihood.__dict__
        if key not in likelihood or not isinstance(likelihood[key], gpflow.base.Parameter):
            raise Exception("parameter name '%s' does not exist" % (key))

        if likelihood[key].shape != val.shape:
            raise Exception("parameter name '%s' must have shape %s and not %s" % (key, likelihood[key].shape, val.shape))

        for i, v in np.ndenumerate(val):
            if v < gpflow.config.default_positive_minimum():
                val[i] = gpflow.config.default_positive_minimum()

        likelihood[key].assign(val)

    def fix_param(self, key):
        """
        Make parameter untrainable (undo with `unfix_param`).

        Args:
            key (str): Name of the parameter.

        Examples:
            >>> model.fix_param('variance')
        """
        if hasattr(self.model.kernel, 'kernels'):
            for kernel_i, kernel in enumerate(self.model.kernel.kernels):
                for param_name, param_val in kernel.__dict__.items():
                    if param_name == key and isinstance(param_val, gpflow.base.Parameter):
                        getattr(self.model.kernel.kernels[kernel_i], param_name).trainable = False
        else:
            for param_name, param_val in self.model.kernel.__dict__.items():
                if param_name == key and isinstance(param_val, gpflow.base.Parameter):
                    getattr(self.model.kernel, param_name).trainable = False

    def unfix_param(self, key):
        """
        Make parameter trainable (that was previously fixed, see `fix_param`).

        Args:
            key (str): Name of the parameter.

        Examples:
            >>> model.unfix_param('variance')
        """
        if hasattr(self.model.kernel, 'kernels'):
            for kernel_i, kernel in enumerate(self.model.kernel.kernels):
                for param_name, param_val in kernel.__dict__.items():
                    if param_name == key and isinstance(param_val, gpflow.base.Parameter):
                        getattr(self.model.kernel.kernels[kernel_i], param_name).trainable = True
        else:
            for param_name, param_val in self.model.kernel.__dict__.items():
                if param_name == key and isinstance(param_val, gpflow.base.Parameter):
                    getattr(self.model.kernel, param_name).trainable = True

    def save_params(self, filename):
        """
        Save model parameters to a given file that can then be loaded with `load_params()`.

        Args:
            filename (str): Filename to save to, automatically appends '.params'.

        Examples:
            >>> model.save_params('filename')
        """
        filename += "." + self.name + ".params"

        try:
            os.remove(filename)
        except OSError:
            pass
        
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)

        data = {
            'model': self.__class__.__name__,
            'likelihood': self.get_likelihood_params(),
            'params': self.get_params()
        }
        with open(filename, 'w') as w:
            json.dump(data, w, cls=NumpyEncoder)

    def load_params(self, filename):
        """
        Load model parameters from a given file that was previously saved with `save_params()`.

        Args:
            filename (str): Filename to load from, automatically appends '.params'.

        Examples:
            >>> model.load_params('filename')
        """
        filename += "." + self.name + ".params"

        with open(filename) as r:
            data = json.load(r)

            if not isinstance(data, dict) or 'model' not in data or 'likelihood' not in data or 'params' not in data:
                raise Exception('parameter file has bad format')
            if not isinstance(data['params'], list) or not all(isinstance(param, dict) for param in data['params']):
                raise Exception('parameter file has bad format')

            if data['model'] != self.__class__.__name__:
                raise Exception("parameter file uses model '%s' which is different from current model '%s'" % (data['model'], self.__class__.__name__))

            cur_params = self.get_params()
            if len(data['params']) != len(cur_params):
                raise Exception("parameter file uses model with %d kernels which is different from current model that uses %d kernels, is the model's Q different?" % (len(data['params']), len(cur_params)))

            for key, val in data['likelihood'].items():
                self.set_likelihood_param(key, val)

            for q, param in enumerate(data['params']):
                for key, val in param.items():
                    self.set_param(q, key, val)

    def train(
        self,
        method='L-BFGS-B',
        tol=1e-6,
        maxiter=2000,
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
            params (dict): Additional dictionary with parameters to minimize. 
            export_graph (bool): Default to False.

        Examples:
            >>> model.train(tol=1e-6, maxiter=1e5)
            
            >>> model.train(method='Adam', opt_params={...})
        """
        #if export_graph:
        #    def get_tensor(name):
        #        return self.graph.get_tensor_by_name('GPR-' + self.model._index + '/likelihood_1/' + name + ':0')
        #    writer = tf.summary.FileWriter("log", self.graph)
        #    K_summary = tf.summary.histogram('K', get_tensor('K'))

        step_i = 0
        def step(theta):
            nonlocal step_i
            #if export_graph:
            #    writer.add_summary(self.session.run(K_summary), step_i)
            step_i += 1

        def loss():
            #x, y = self.model.data
            #K = self.model.kernel(x)
            #num_data = x.shape[0]
            #k_diag = tf.linalg.diag_part(K)
            #s_diag = tf.fill([num_data], self.model.likelihood.variance)
            #ks = tf.linalg.set_diag(K, k_diag + s_diag)
            #tf.debugging.check_numerics(ks, "ks")
            #if not np.all(np.linalg.eigvals(ks) > 0):
            #    print("NOT POSITIVE DEFINITE:")
            #    print(ks)
            return -self.model.log_marginal_likelihood()

        start_time = time.time()
        if method.lower() == "adam":
            opt = tf.optimizers.Adam(learning_rate=0.001)
            opt.minimize(loss, self.model.trainable_variables)
        else:
            opt = gpflow.optimizers.Scipy()
            opt.minimize(closure=loss, variables=self.model.trainable_variables, method=method, tol=tol, options={'maxiter': maxiter, 'disp': False}, **params)

        print("Done in %.1f minutes" % ((time.time() - start_time)/60))

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

        Examples:
            >>> model.predict(plot=True)
        """
        if x is not None:
            self.dataset.set_pred(x)

        x = self.dataset.to_kernel_pred()
        if len(x) == 0:
            raise Exception('no prediction x range set, use pred_x argument or set manually using DataSet.set_pred() or Data.set_pred()')

        mu, var = self.model.predict_f(x)
        self.dataset.from_kernel_pred(self.name, mu, var)
        
        if plot:
            self.plot_prediction()

        _, mu, lower, upper = self.dataset.get_pred(self.name)
        return mu, lower, upper

    # TODO
    def plot_prediction(self, grid=None, figsize=(12, 8), ylims=None, names=None, title=''):

        """
        Plot training points, all data and prediction for training range for all channels.

        Args:
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
            axes[i].plot(x_train[i][:,0], y_train[i], '.k', label='Train', ms=5)
            axes[i].plot(x_all[i][:,0], y_all[i], '--', label='Test', c='gray',lw=1.4, zorder=5)
            
            axes[i].plot(x_pred[i][:,0], mu[i], label='Post.Mean', c=colors[i], zorder=1)
            axes[i].fill_between(x_pred[i][:,0].reshape(-1),
                lower[i],
                upper[i],
                label='95% c.i',
                color=colors[i],
                alpha=0.4)
            
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
            
        plt.suptitle(title, y=1.02, fontsize=20)
        plt.tight_layout()
        
        return fig, axes

