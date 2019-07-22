from .data import Data
from .model import model
from .kernels import SpectralMixture, sm_init, SpectralMixtureOLD, MultiOutputSpectralMixture, SpectralMixtureLMC, ConvolutionalGaussian, CrossSpectralMixture
import numpy as np

def transform_multioutput_data(x, y=None):
    chan = []
    if isinstance(x, list):
        for channel in range(len(x)):
            chan.append(channel * np.ones(len(x[channel])))
        chan = np.concatenate(chan)
    elif isinstance(x, dict):
        for channel, data in x.items():
            chan.append(channel * np.ones(len(data)))
        chan = np.concatenate(chan)
        x = np.array(list(x.values()))
    else:
        raise Exception("unknown data type for x")
    
    x = np.concatenate(x)
    x = np.stack((chan, x), axis=1)
    if y != None:
        y = np.concatenate(y).reshape(-1, 1)
    return x, y

def estimate_from_sm(data, Q):
    """ returns format: params[q][name][input dim][channel]"""
    params = []
    for q in range(Q):
        params.append({'weight': [], 'mean': [], 'scale': []})

    for channel in range(data.get_output_dims()):
        sm_data = Data()
        sm_data.add(data.X[channel], data.Y[channel])

        sm = SM(sm_data, Q)
        sm.estimate_from_bnse()
        sm.train(method='BFGS', disp=True)

        for q in range(Q):
            params[q]['weight'].append(sm.params[q]['mixture_weights']),
            params[q]['mean'].append(sm.params[q]['mixture_means']),
            params[q]['scale'].append(sm.params[q]['mixture_scales'].T),

    for q in range(Q):
        params[q]['weight'] = np.array(params[q]['weight'])
        params[q]['mean'] = np.array(params[q]['mean']) * np.pi * 2
        params[q]['scale'] = np.array(params[q]['scale'])
    return params

class SM(model):
    """SM is the standard Spectral Mixture kernel."""
    def __init__(self, data, Q=1, name="SM"):
        model.__init__(self, name, data, Q)

        input_dims = self.data.get_input_dims()
        output_dims = self.data.get_output_dims()
        if output_dims != 1:
            raise Exception("Single output Spectral Mixture kernel can only take one output dimension in the data")

        x, y = self._transform_data(data.X, data.Y)
        weights, means, scales = sm_init(x, y, Q)
        for q in range(Q):
            self.params.append({
                'mixture_weights': weights[q],
                'mixture_means': np.array(means[q]),
                'mixture_scales': np.array(scales.T[q]),
            })

    def estimate_from_bnse(self):
        means, amplitudes = self.data.get_bnse_estimation(self.Q)
        weights = amplitudes[0] * self.data.Y[0].std()
        weights = np.sqrt(weights/np.sum(weights))

        for q in range(self.Q):
            self.params[q]['mixture_weights'] = weights[0][q]
            self.params[q]['mixture_means'] = means[0].T[q] / np.pi / 2.0

    def _transform_data(self, x, y=None):
        if isinstance(x, dict):
            x = np.array(list(x.values()))
        elif isinstance(x, list):
            x = np.array(x)
        y = np.array(y)
        return x.T, y.T

    def _kernel(self):
        weights = np.array([self.params[q]['mixture_weights'] for q in range(self.Q)])
        means = np.array([self.params[q]['mixture_means'] for q in range(self.Q)])
        scales = np.array([self.params[q]['mixture_scales'] for q in range(self.Q)]).T
        return SpectralMixture(
            self.Q,
            weights,
            scales,
            means,
            self.data.get_input_dims()
        )

    def _update_params(self, trainables):
        for key, val in trainables.items():
            names = key.split("/")
            if len(names) == 3 and names[1] == 'kern':
                name = names[2]
                if name == 'mixture_scales':
                    val = val.T
                for q in range(len(val)):
                    self.params[q][name] = val[q]

class MOSM(model):
    """
    Multi Output Spectral Mixture kernel as proposed by our paper.

    It takes a number of components Q and allows for recommended initial
    parameter estimation to improve optimization outputs."""
    def __init__(self, data, Q=1, name="MOSM"):
        model.__init__(self, name, data, Q)

        input_dims = self.data.get_input_dims()
        output_dims = self.data.get_output_dims()

        for _ in range(Q):
            self.params.append({
                "magnitude": np.random.standard_normal((output_dims)),
                "mean": np.random.standard_normal((input_dims, output_dims)),
                "variance": np.random.random((input_dims, output_dims)),
                "delay": np.zeros((input_dims, output_dims)),
                "phase": np.zeros((output_dims)),
                "noise": np.random.random((output_dims)),
            })

    def estimate_from_bnse(self):
        peaks, _ = self.data.get_bnse_estimation(self.Q)
        for channel in range(self.data.get_output_dims()):
            for q in range(self.Q):
                self.params[q]["mean"] = peaks[channel].T[q].reshape(-1, 1)

    def estimate_from_sm(self):
        params = estimate_from_sm(self.data, self.Q)
        for q in range(self.Q):
            self.params[q]["magnitude"] = params[q]['weight']
            self.params[q]["mean"] = params[q]['mean']
            self.params[q]["variance"] = params[q]['scale']

    def _transform_data(self, x, y=None):
        return transform_multioutput_data(x, y)

    def _kernel(self):
        for q in range(self.Q):
            kernel = MultiOutputSpectralMixture(
                self.data.get_input_dims(),
                self.data.get_output_dims(),
                self.params[q]["magnitude"],
                self.params[q]["mean"],
                self.params[q]["variance"],
                self.params[q]["delay"],
                self.params[q]["phase"],
                self.params[q]["noise"],
            )

            if q == 0:
                kernel_set = kernel
            else:
                kernel_set += kernel
        return kernel_set

class CSM(model):
    """
    Cross Spectral Mixture kernel with Q components and Rq latent functions
    (TODO: true?).
    """
    def __init__(self, data, Q=1, Rq=1, name="CSM"):
        model.__init__(self, name, data, Q)
        self.Rq = Rq

    def _transform_data(self, x, y=None):
        return transform_multioutput_data(x, y)

    def _kernel(self):
        for q in range(self.Q):
            kernel = CrossSpectralMixture(self.data.get_input_dims(), self.data.get_output_dims(), self.Rq)
            if q == 0:
                kernel_set = kernel
            else:
                kernel_set += kernel
        return kernel_set

class SM_LMC(model):
    """
    Spectral Mixture - Linear Model of Coregionalization kernel with Q components and Rq latent functions
    (TODO: true?)."""
    def __init__(self, data, Q=1, Rq=1, name="SM-LMC"):
        model.__init__(self, name, data, Q)
        self.Rq = Rq

    def _transform_data(self, x, y=None):
        return transform_multioutput_data(x, y)

    def _kernel(self):
        for q in range(self.Q):
            kernel = SpectralMixtureLMC(self.data.get_input_dims(), self.data.get_output_dims(), self.Rq)
            if q == 0:
                kernel_set = kernel
            else:
                kernel_set += kernel
        return kernel_set

class CG(model):
    """
    CG is the Convolutional Gaussian kernel with Q components.
    """
    def __init__(self, data, Q=1, name="CG"):
        model.__init__(self, name, data, Q)

    def _transform_data(self, x, y=[]):
        return transform_multioutput_data(x, y)

    def _kernel(self):
        for q in range(self.Q):
            kernel = ConvolutionalGaussian(self.data.get_input_dims(), self.data.get_output_dims())
            if q == 0:
                kernel_set = kernel
            else:
                kernel_set += kernel
        return kernel_set
    
