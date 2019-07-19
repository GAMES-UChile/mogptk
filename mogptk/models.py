from .data import Data
from .model import model
from .kernels import SpectralMixture, sm_init, SpectralMixtureOLD, MultiOutputSpectralMixture, SpectralMixtureLMC, ConvolutionalGaussian, CrossSpectralMixture
import numpy as np

def mo_transform_data(x, y=None):
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
        self.parameters = [{
            "mixture_weights": weights,
            "mixture_scales": scales,
            "mixture_means": means,
        }]

    def _transform_data(self, x, y=None):
        if isinstance(x, dict):
            x = np.array(list(x.values()))
        elif isinstance(x, list):
            x = np.array(x)
        y = np.array(y)
        return x.T, y.T

    def _kernel(self):
        param = self.parameters[0]
        return SpectralMixture(self.Q, param['mixture_weights'], param['mixture_scales'], param['mixture_means'], self.data.get_input_dims())

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
            self.parameters.append({
                "magnitude": np.random.randn(output_dims),
                "mean": np.random.randn(input_dims, output_dims),
                "variance": np.random.random((input_dims, output_dims)),
                "delay": np.zeros((input_dims, output_dims)),
                "phase": np.zeros(output_dims),
                "noise": np.random.random((output_dims)),
            })

    def estimate_means(self):
        peaks, _ = self.data.get_bnse_estimation(self.data.get_output_dims())
        for q in range(self.Q):
            if self.parameters[q]["mean"].shape == peaks[q].shape:
                self.parameters[q]["mean"] = peaks[q]

    def estimate_params(self):
        means, amplitudes = self.data.get_bnse_estimation(self.data.get_output_dims())

        for channel in range(self.data.get_output_dims()):
            data = Data()
            data.add(self.data.X[channel], self.data.Y[channel])

            weights = amplitudes[channel] * self.data.Y[channel].std()
            weights = np.sqrt(weights/np.sum(weights))

            sm = SM(data, self.Q)
            sm.train(method='BFGS', disp=True)
            
            for q in range(self.Q):
                weights = sm.parameters[0]['mixture_weights']
                means = sm.parameters[0]['mixture_means']
                scales = sm.parameters[0]['mixture_scales']
                for i in range(means.shape[1]):
                    self.parameters[q]["magnitude"][channel] = weights[q]
                    self.parameters[q]["mean"][i][channel] = means[q][i]
                    self.parameters[q]["variance"][i][channel] = scales[i][q]

    def _transform_data(self, x, y=None):
        return mo_transform_data(x, y)

    def _kernel(self):
        params = self.parameters
        for q in range(self.Q):
            kernel = MultiOutputSpectralMixture(
                self.data.get_input_dims(),
                self.data.get_output_dims(),
                params[q]["magnitude"],
                params[q]["mean"],
                params[q]["variance"],
                params[q]["delay"],
                params[q]["phase"],
                params[q]["noise"],
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
        return mo_transform_data(x, y)

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
        return mo_transform_data(x, y)

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
        return mo_transform_data(x, y)

    def _kernel(self):
        for q in range(self.Q):
            kernel = ConvolutionalGaussian(self.data.get_input_dims(), self.data.get_output_dims())
            if q == 0:
                kernel_set = kernel
            else:
                kernel_set += kernel
        return kernel_set
    
