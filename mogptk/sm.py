import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

from .dataset import DataSet
from .model import Model, Exact, logger
from .kernels import SpectralKernel, IndependentMultiOutputKernel, MixtureKernel, positive_minimum
from .plot import plot_spectrum

class SM(Model):
    """
    An independent spectral mixture kernel as proposed by [1] per channel. The parameters will be randomly instantiated, use init_parameters() to initialize the parameters to reasonable values for the current dataset.

    Args:
        dataset (mogptk.dataset.DataSet): DataSet object of data for all channels.
        Q (int, optional): Number of components.
        model: Gaussian Process model to use, such as mogptk.Exact.
        name (str, optional): Name of the model.

    Attributes:
        dataset: The associated mogptk.DataSet.
        model: The mogptk.kernels.Model.
        kernel: The mogptk.kernels.Kernel.

    Examples:

    >>> import numpy as np
    >>> import mogptk
    >>> 
    >>> t = np.linspace(0, 10, 100)
    >>> y = np.sin(0.5 * t)
    >>> 
    >>> data = mogptk.Data(t, y)
    >>> model = mogptk.SM(data, Q=1)
    >>> model.init_parameters()
    >>> model.train()
    >>> model.predict()
    >>> data.plot()

    [1] A.G. Wilson and R.P. Adams, "Gaussian Process Kernels for Pattern Discovery and Extrapolation", International Conference on Machine Learning 30, 2013
    """
    def __init__(self, dataset, Q=1, model=Exact(), name="SM"):
        if not isinstance(dataset, DataSet):
            dataset = DataSet(dataset)

        kernel = IndependentMultiOutputKernel(
            [MixtureKernel(SpectralKernel(dataset[i].get_input_dims()), Q) for i in range(dataset.get_output_dims())],
            output_dims=dataset.get_output_dims(),
        )

        super(SM, self).__init__(dataset, kernel, model, name)
        self.Q = Q
        if issubclass(type(model), Exact):
            self.model.noise.assign(0.0, lower=0.0, trainable=False)  # handled by MultiOutputKernel

    def init_parameters(self, method='BNSE', noise=False):
        """
        Initialize parameters of kernel from the data.

        Arguments:
            method (str, optional): Method of estimation, such as IPS, LS, or BNSE.
            noise (boolean, optional): Add noise of std.dev. equal to 1/10th of the estimated value.

        Methods:
            IPS: Independent parameter sampling (from the PhD thesis of Andrew Wilson 2014) takes the inverse of lengthscales drawn from truncated Gaussian N(0, max_dist^2), the means drawn from Unif(0, 0.5 / minimum distance between two points), and the mixture weights by taking the standard variation of the y values divided by the number of mixtures.
            LS: Uses Lomb Scargle periodogram for estimating the PSD, and using the first Q peaks as the means and mixture weights.
            BNSE: Uses the BNSE (Tobar 2018) to estimate the PSD and use the first Q peaks as the means and mixture weights.
        """

        input_dims = self.dataset.get_input_dims()
        output_dims = self.dataset.get_output_dims()

        if method.lower() not in ['ips', 'ls', 'bnse']:
            raise ValueError("valid methods of estimation are IPS, LS, and BNSE")

        if method.lower() == 'ips':
            for j in range(output_dims):
                nyquist = self.dataset[j].get_nyquist_estimation()
                x, y = self.dataset[j].get_train_data(transformed=True)
                x_range = np.max(x, axis=0) - np.min(x, axis=0)

                weights = [y.std()/self.Q] * self.Q
                means = nyquist * np.random.rand(self.Q, input_dims[j])
                variances = 1.0 / (np.abs(np.random.randn(self.Q, input_dims[j])) * x_range)

                for q in range(self.Q):
                    self.model.kernel[j][q].weight.assign(weights[q])
                    self.model.kernel[j][q].mean.assign(means[q,:])
                    self.model.kernel[j][q].variance.assign(variances[q,:])
            return
        elif method.lower() == 'ls':
            amplitudes, means, variances = self.dataset.get_lombscargle_estimation(self.Q)
            if len(amplitudes) == 0:
                logger.warning('LS could not find peaks for SM')
                return
        elif method.lower() == 'bnse':
            amplitudes, means, variances = self.dataset.get_bnse_estimation(self.Q)
            if np.sum(amplitudes) == 0.0:
                logger.warning('BNSE could not find peaks for SM')
                return

        #if noise:
        #    # TODO: what is this? check noise for all kernels
        #    pct = 1/30.0
        #    # noise proportional to the values
        #    noise_amp = np.random.multivariate_normal(
        #        mean=np.zeros(self.Q),
        #        cov=np.diag(amplitudes.mean(axis=1) * pct))
        #    # set value to a minimun value
        #    amplitudes = np.maximum(np.zeros_like(amplitudes) + 1e-6, amplitudes + noise_amp)

        #    noise_mean = np.random.multivariate_normal(
        #        mean=np.zeros(self.Q),
        #        cov=np.diag(means.mean(axis=1) * pct))
        #    means = np.maximum(np.zeros_like(means) + 1e-6, means + noise_mean)

        #    noise_var = np.random.multivariate_normal(
        #        mean=np.zeros(self.Q),
        #        cov=np.diag(variances.mean(axis=1) * pct))
        #    variances = np.maximum(np.zeros_like(variances) + 1e-6, variances + noise_var)

        for j in range(output_dims):
            # TODO: check weights for all kernels
            _, y = self.dataset[j].get_train_data(transformed=True)
            mixture_weights = amplitudes[j].mean(axis=1)
            if 0.0 < amplitudes[j].sum():
                mixture_weights /= amplitudes[j].sum()
            mixture_weights *= 2.0 * y.std()
            for q in range(self.Q):
                self.model.kernel[j][q].weight.assign(mixture_weights[q])
                self.model.kernel[j][q].mean.assign(means[j][q,:])
                self.model.kernel[j][q].variance.assign(variances[j][q,:])

    def plot_spectrum(self, title=None):
        """
        Plot spectrum of kernel.
        """
        names = self.dataset.get_names()
        nyquist = self.dataset.get_nyquist_estimation()
        output_dims = self.dataset.get_output_dims()

        means = np.array([[self.model.kernel[q][j].mean.numpy() for j in range(output_dims)] for q in range(self.Q)])
        scales = np.array([[self.model.kernel[q][j].variance.numpy() for j in range(output_dims)] for q in range(self.Q)])
        weights = np.array([[self.model.kernel[q][j].weight.numpy()[0] for j in range(output_dims)] for q in range(self.Q)])

        return plot_spectrum(means, scales, weights=weights, nyquist=nyquist, titles=names, title=title)
