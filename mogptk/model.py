import numpy as np
import gpflow
from sklearn import metrics

import logging
logging.getLogger('tensorflow').propagate = False

def load(filename):
    # data?
    m = gpflow.saver.Saver().load(filename)
    return model(None, m)

def Model(data, kernel):
    x, y = data.get_ts_observations()
    m = gpflow.models.GPR(x, y, kernel.build())
    return model(data, m, kernel.name)

def SparseModel(data, kernel):
    x, y = data.get_ts_observations()
    m = gpflow.models.SGPR(x, y, kernel.build())
    return model(data, m, kernel.name + " (sparse)")

def SparseVariationalModel(data, kernel):
    x, y = data.get_ts_observations()
    m = gpflow.models.SVGP(x, y, kernel.build(), gpflow.likelihoods.Gaussian())
    return model(data, m, kernel.name + " (sparse-variational)")

class model:
    def __init__(self, data, model, name=""):
        self.name = name
        self.data = data
        self.model = model
        self.X_pred = {}
        self.Y_mu_pred = {}
        self.Y_var_pred = {}

    def save(self, filename):
        gpflow.saver.Saver().save(filename, self.model)

    def set_prediction_range(self, channel, start=None, end=None, step=None, n=None):
        channel = self.data.get_channel_index(channel)
        if start == None:
            start = self.data.X[channel][0]
        if end == None:
            end = self.data.X[channel][-1]
        if end <= start:
            raise Exception("start must be lower than end")
        if step == None and n == None:
            step = (end-start)/100
        elif step == None:
            step = (end-start)/n

        self.X_pred[channel] = np.arange(start, end+step, step)

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

def errors(*args, **kwargs):
    all_obs = False
    if "all_obs" in kwargs:
        all_obs = kwargs["all_obs"]
    output = False
    if "print" in kwargs:
        output = kwargs["print"]

    errors = {
        "model": [],
        "MAE": [],
        "MSE": [],
    }
    for model in args:
        Y_true = np.empty(0)
        Y_pred = np.empty(0)
        for channel in range(model.data.get_output_dimensions()):
            if all_obs:
                x, y_true = model.data.get_all_observations(channel)
            else:
                x, y_true = model.data.get_deleted_observations(channel)

            if len(x) > 0:
                y_pred = np.interp(x, model.X_pred[channel], model.Y_mu_pred[channel])

                Y_true = np.append(Y_true, y_true)
                Y_pred = np.append(Y_pred, y_pred)
            
        errors["model"].append(model.name)
        errors["MAE"].append(metrics.mean_absolute_error(Y_true, Y_pred))
        errors["MSE"].append(metrics.mean_squared_error(Y_true, Y_pred))

    if output:
        import pandas as pd
        df = pd.DataFrame(errors)
        df.set_index('model', inplace=True)
        display(df)
    else:
        return errors

