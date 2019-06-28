import numpy as np
import gpflow

def load(filename):
    # data?
    m = gpflow.saver.Saver().load(filename)
    return model(None, m)

def Model(data, kernel):
    x, y = data.get_observations()
    m = gpflow.models.GPR(x, y, kernel.build())
    return model(data, m)

def SparseModel(data, kernel):
    x, y = data.get_observations()
    m = gpflow.models.SGPR(x, y, kernel.build())
    return model(data, m)

def SparseVariationalModel(data, kernel):
    x, y = data.get_observations()
    m = gpflow.models.SVGP(x, y, kernel.build(), gpflow.likelihoods.Gaussian())
    return model(data, m)

class model:
    def __init__(self, data, model):
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

    def optimize(self, optimizer='L-BFGS-B', maxiter=1000, disp=True):
        opt = gpflow.train.ScipyOptimizer(method=optimizer)
        opt_tensor = opt.make_optimize_tensor(self.model, gpflow.get_default_session(), maxiter=maxiter, disp=disp)
        opt_tensor.minimize(session=gpflow.get_default_session())

        # TODO: AdamOptimizer from gpflow.training

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

