from .model import model
from .kernels import MultiOutputSpectralMixture, SpectralMixtureLMC, ConvolutionalGaussian, CrossSpectralMixture
import numpy as np

class MOSM(model):
    """MOSM is the Multi Output Spectral Mixture kernel as proposed by our paper. It takes a number of components Q and allows for recommended initial parameter estimation to improve optimization outputs."""
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
        peaks, _ = self.data.get_bnse_estimation(self.Q)
        for q in range(self.Q):
            self.parameters[q]["mean"] = peaks[q]

    def _kernel(self):
        params = self.parameters
        for q in range(self.Q):
            kernel = MultiOutputSpectralMixture(self.data.get_input_dims(), self.data.get_output_dims(), params[q]["magnitude"], params[q]["mean"], params[q]["variance"], params[q]["delay"], params[q]["phase"], params[q]["noise"])
            if q == 0:
                kernel_set = kernel
            else:
                kernel_set += kernel
        return kernel_set

    def _update_parameters(self, trainables):
        for key, val in trainables.items():
            names = key.split("/", 4)
            if len(names) == 5 and names[1] == 'kern' and names[2] == 'kernels':
                q = int(names[3])
                name = names[4]
                self.parameters[q][name] = val

class CSM(model):
    """CSM is the Cross Spectral Mixture kernel with Q components and Rq latent functions (TODO: true?)."""
    def __init__(self, data, Q=1, Rq=1, name="CSM"):
        model.__init__(self, name, data, Q)
        self.Rq = Rq

    def _kernel(self):
        for q in range(self.Q):
            kernel = CrossSpectralMixture(self.data.get_input_dims(), self.data.get_output_dims(), self.Rq)
            if q == 0:
                kernel_set = kernel
            else:
                kernel_set += kernel
        return kernel_set
    
    def _update_parameters(self, trainables):
        pass

class SM_LMC(model):
    """SM_LMC is the Spectral Mixture - Linear Model of Coregionalization kernel with Q components and Rq latent functions (TODO: true?)."""
    def __init__(self, data, Q=1, Rq=1, name="SM-LMC"):
        model.__init__(self, name, data, Q)
        self.Rq = Rq

    def _kernel(self):
        for q in range(self.Q):
            kernel = SpectralMixtureLMC(self.data.get_input_dims(), self.data.get_output_dims(), self.Rq)
            if q == 0:
                kernel_set = kernel
            else:
                kernel_set += kernel
        return kernel_set
    
    def _update_parameters(self, trainables):
        pass

class CG(model):
    """CG is the Convolutional Gaussian kernel with Q components."""
    def __init__(self, data, Q=1, name="CG"):
        model.__init__(self, name, data, Q)

    def _kernel(self):
        for q in range(self.Q):
            kernel = ConvolutionalGaussian(self.data.get_input_dims(), self.data.get_output_dims())
            if q == 0:
                kernel_set = kernel
            else:
                kernel_set += kernel
        return kernel_set
    
    def _update_parameters(self, trainables):
        pass

