import os
import numpy as np
from pprint import pprint
import gpflow
from .tf import show_default_graph
import tensorflow as tf

import logging
logging.getLogger('tensorflow').propagate = False

class model:
    """
    Multioutput Gaussian proccess model. Can be either MOSM, CSM, SM-LMC or CONV.

    This class is used to use a multioutput GP model, train and test data can be added,
    the model can be used to train a predict.

    Example:

        ---TO-DO---

    Atributes:
        name ():
        data (obj, instance of mogptk.data):
        model ():
        Q (int): Number of components of the model.
        parameters ():
    """

    def __init__(self, name, data, Q):
        self.name = name
        self.data = data
        self.model = None
        self.Q = Q
        self.params = []
        self.fixed_params = []
        self.X_pred = {}
        self.Y_mu_pred = {}
        self.Y_var_pred = {}

    def _kernel(self):
        # overridden by specific models
        raise Exception("kernel not specified")
    
    def _transform_data(self, x, y):
        # overridden by specific models
        raise Exception("kernel not specified")

    def _update_params(self, trainables):
        for key, val in trainables.items():
            names = key.split("/")
            if len(names) == 5 and names[1] == 'kern' and names[2] == 'kernels':
                q = int(names[3])
                name = names[4]
                self.params[q][name] = val

    def get_params(self):
        """
        Returns all parameters set for the kernel per component.
        """
        return self.params
        
    def _get_param_across(self, name='mixture_means'):
        """
        Get all the name parameters across all components
        """
        return np.array([self.params[q][name] for q in range(self.Q)])

    def set_param(self, q, key, val):
        """
        Sets an initial kernel parameter prior to optimizations for component q with key the parameter name.

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
        self.model.mogptk_data = self.data._encode()
        self.model.mogptk_Q = self.Q
        self.model.mogptk_params = self.params
        self.model.mogptk_fixed_params = self.fixed_params

        with self.graph.as_default():
            with self.session.as_default():
                gpflow.Saver().save(filename, self.model)

    def build(self, kind='full', disp=True):
        if disp:
            print("Building...")

        x, y = self._transform_data(self.data.X, self.data.Y)

        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)
        with self.graph.as_default():
            with self.session.as_default():
                if kind == 'full':
                    self.model = gpflow.models.GPR(x, y, self._kernel())
                elif kind == 'sparse':
                    # TODO: test if induction points are set
                    self.name += ' (sparse)'
                    self.model = gpflow.models.SGPR(x, y, self._kernel())
                elif kind == 'sparse-variational':
                    self.name += ' (sparse-variational)'
                    self.model = gpflow.models.SVGP(x, y, self._kernel(), gpflow.likelihoods.Gaussian())
                else:
                    raise Exception("model type '%s' does not exist" % (kind))
        
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
        
        self.X_pred = {}
        self.Y_mu_pred = {}
        self.Y_var_pred = {}

    def train(self, method='L-BFGS-B', kind='full', maxiter=2000, disp=False, learning_rate=0.001, export_graph=False):
        """
        Builds and trains the model using the kernel and its parameters.

        For different optimizers, see scipy.optimize.minimize.
        It can be bounded by a maximum number of iterations, disp will output final
        optimization information.
        When using the 'Adam' optimizer, a learning_rate can be set.

        Args:
            kind (str): Type of model to use, posible mode are 'full', 'sparse' and
                'sparse-variational'.

            method (str): Optimizer to use, if "Adam" is chosen,
                gpflow.training.Adamoptimizer will be used, otherwise the passed scipy
                optimizer is used. Default to scipy 'L-BFGS-B'.

            maxiter (int): Maximum number of iterations, default to 1000.

            disp (bool): If true it display information on the optimization, only valid
                to scipy optimizers. Default to True.

            disp_graph (bool): If true show computational graph.
                Requires tensorboard and graphviz installed.

            learning_rate(float): Learning rate of Adam optimizer.
                Only valid with Adam, default to 0.001.
        """
        
        self.build(kind, disp)

        if disp:
            print("Optimizing...")

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
                    opt = gpflow.training.AdamOptimizer(learning_rate)
                    opt.minimize(self.model, anchor=True, maxiter=maxiter)
                else:
                    opt = gpflow.train.ScipyOptimizer(method=method)
                    opt.minimize(self.model, anchor=True, disp=disp, maxiter=maxiter, step_callback=step)

                self._update_params(self.model.read_trainables())

                if disp:
                    print("Done")


    ################################################################################
    # Predictions ##################################################################
    ################################################################################

    def predict(self, X_pred, disp_graph=False):
        """
        Predict with model.

        Will make a prediction using x as input. If no input value is passed, the prediction will 
        be made with atribute self.X_pred that can be setted with other functions.
        It returns the X, Y_mu, Y_var values per channel.

        Args:
            x (dict): Dictionary where keys are channel index and elements numpy arrays with 
                          channel inputs.

        Returns:
            X_pred, Y_mu_pred, Y_var_pred: Prediction input, output and variance of the model.

        """
        if self.model == None:
            raise Exception("build (and train) the model before doing predictions")

        with self.graph.as_default():
            with self.session.as_default():
                x, _ = self._transform_data(X_pred)
                mu, var = self.model.predict_f(x)

                if disp_graph:
                    show_default_graph()

        n = 0
        for channel in X_pred:
            n += X_pred[channel].shape[0]
        
        i = 0
        Y_mu_pred = {}
        Y_var_pred = {}
        for channel in X_pred:
            n = X_pred[channel].shape[0]
            if n != 0:
                Y_mu_pred[channel] = mu[i:i+n].reshape(1, -1)[0]
                Y_var_pred[channel] = var[i:i+n].reshape(1, -1)[0]
                i += n
        return Y_mu_pred, Y_var_pred
