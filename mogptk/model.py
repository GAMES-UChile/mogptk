import numpy as np
import gpflow

import logging
logging.getLogger('tensorflow').propagate = False

def load(filename):
    # TODO: load training data too?
    m = gpflow.saver.Saver().load(filename)
    return model(None, m)

class model:
    def __init__(self, name, data, Q):
        self.name = name
        self.data = data
        self.model = None
        self.Q = Q
        self.parameters = []

    def set_parameter(self, q, key, val):
        """set_parameter sets an initial kernel parameter prior to optimizations for component q with key the parameter name."""
        if q < 0 or len(self.parameters) <= q:
            raise Exception("qth component %d does not exist" % (q))
        if key not in self.parameters[q]:
            raise Exception("parameter name '%s' does not exist" % (key))
        self.parameters[q][key] = val

    def kernel(self):
        raise Exception("kernel not specified")

    def build(self, kind='full'):
        """build builds the model using the kernel and its parameters."""
        x, y = self.data.get_ts_obs()
        if kind == 'full':
            self.model = gpflow.models.GPR(x, y, self.kernel())
        elif kind == 'sparse':
            # TODO: test if induction points are set
            self.name += ' (sparse)'
            self.model = gpflow.models.SGPR(x, y, self.kernel())
        elif kind == 'sparse-variational':
            self.name += ' (sparse-variational)'
            self.model = gpflow.models.SVGP(x, y, self.kernel(), gpflow.likelihoods.Gaussian())
        else:
            raise Exception("model type '%s' does not exist" % (kind))

        self.X_pred = {}
        self.Y_mu_pred = {}
        self.Y_var_pred = {}

    def save(self, filename):
        # TODO: needs testing and saving training data?
        gpflow.saver.Saver().save(filename, self.model)

    def optimize(self, optimizer='L-BFGS-B', maxiter=1000, disp=True, learning_rate=0.001):
        """optimize optimizes the kernel parameters by an optimizer (see scipy.optimize.minimize for available optimizers). It can be bounded by a maximum number of iterations, disp will output final optimization information. When using the 'Adam' optimizer, a learning_rate can be set."""
        if self.model == None:
            raise Exception("build the model before optimizing it")

        if optimizer == 'Adam':
            from gpflow.actions import Loop, Action
            from gpflow.training import AdamOptimizer

            adam = AdamOptimizer(learning_rate).make_optimize_action(self.model)
            Loop([adam], stop=maxiter)()
        else:
            session = gpflow.get_default_session()
            opt = gpflow.train.ScipyOptimizer(method=optimizer)
            opt_tensor = opt.make_optimize_tensor(self.model, session, maxiter=maxiter, disp=disp)
            opt_tensor.minimize(session=session)

            #self.model.anchor(session)
            #print(self.model.read_trainables())


    ################################################################################
    # Predictions ##################################################################
    ################################################################################

    def get_predictions(self):
        """get_predictions returns the X, Y_mu, Y_var values per channel."""
        if len(self.X_pred) == 0:
            raise Exception("use predict before retrieving the predictions on the model")
        return self.X_pred, self.Y_mu_pred, self.Y_var_pred

    def get_channel_predictions(self, channel):
        """get_channel_predictions returns the X, Y_mu, Y_var values for a certain channel."""
        if len(self.X_pred) == 0:
            raise Exception("use predict before retrieving the predictions on the model")
        channel = self.data.get_channel_index(channel)
        return self.X_pred[channel], self.Y_mu_pred[channel], self.Y_var_pred[channel]

    def set_prediction_range(self, channel, start=None, end=None, step=None, n=None):
        """set_prediction_range sets the prediction range for a certain channel in the interval [start,end] with either a stepsize step or a number of points n."""
        channel = self.data.get_channel_index(channel)
        if start == None:
            start = self.data.X[channel][0]
        if end == None:
            end = self.data.X[channel][-1]
        if end <= start:
            raise Exception("start must be lower than end")

        if step == None and n != None:
            self.X_pred[channel] = np.linspace(start, end, n)
        else:
            if step == None:
                step = (end-start)/100
            self.X_pred[channel] = np.arange(start, end+step, step)

    def set_prediction_x(self, channel, x):
        """set_prediction_x sets the prediction range using a list of Numpy array for a certain channel."""
        channel = self.data.get_channel_index(channel)
        if isinstance(x, list):
            x = np.array(x)
        elif not isinstance(x, np.ndarray):
            raise Exception("x expected to be a list or Numpy array")

        self.X_pred[channel] = x

    def predict(self):
        """predict will make a prediction in the data ranges set by the various prediction range setters. This function will return the X, Y_mu, Y_var values per channel."""
        if self.model == None:
            raise Exception("build (and optimize) the model before doing predictions")

        chan = []
        for channel in self.X_pred:
            chan.append(channel * np.ones(len(self.X_pred[channel])))
        chan = np.concatenate(chan)
        x = np.concatenate(list(self.X_pred.values()))
        x = np.stack((chan, x), axis=1)

        mu, var = self.model.predict_f(x)

        n = 0
        for channel in self.X_pred:
            n += self.X_pred[channel].shape[0]
        
        i = 0
        for channel in self.X_pred:
            n = self.X_pred[channel].shape[0]
            if n != 0:
                self.Y_mu_pred[channel] = mu[i:i+n].reshape(1, -1)[0]
                self.Y_var_pred[channel] = var[i:i+n].reshape(1, -1)[0]
                i += n
        return self.X_pred, self.Y_mu_pred, self.Y_var_pred

