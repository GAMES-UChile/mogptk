import numpy as np
import gpflow

import logging
logging.getLogger('tensorflow').propagate = False

def load(filename):
    # TODO: data?
    m = gpflow.saver.Saver().load(filename)
    return model(None, m)

class model:
    def __init__(self, name, data, Q):
        self.name = name
        self.data = data
        self.model = None
        self.Q = Q
        self.parameters = []

        self.X_pred = {}
        self.Y_mu_pred = {}
        self.Y_var_pred = {}

    def set_parameter(self, q, key, val):
        if q < 0 or len(self.parameters) <= q:
            raise Exception("qth component does not exist")
        if key not in self.parameters[q]:
            raise Exception("parameter '%s' does not exist" % (key))
        self.parameters[q][key] = val

    def kernel(self):
        raise Exception("kernel not specified")

    def build(self, kind='regular'):
        x, y = self.data.get_ts_obs()
        if kind == 'regular':
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

    def save(self, filename):
        gpflow.saver.Saver().save(filename, self.model)

    def optimize(self, optimizer='L-BFGS-B', maxiter=1000, disp=True, learning_rate=0.001):
        if optimizer == 'adam':
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

    def set_prediction_range(self, channel, start=None, end=None, step=None, n=None):
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
        channel = self.data.get_channel_index(channel)
        if isinstance(x, list):
            x = np.array(x)
        elif not isinstance(x, np.ndarray):
            raise Exception("x expected to be a list or Numpy array")

        self.X_pred[channel] = x

    def predict(self):
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

