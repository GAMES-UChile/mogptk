import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import logging

from .model import model, logger
from .kernels import SpectralKernel, MixtureKernel, positive_minimum

class SM(model):
    """
    A single output GP Spectral mixture kernel as proposed by [1].

    The model contain the dataset and the associated gpflow model, 
    when the mogptk.Model is instanciated the gpflow model is built 
    using random parameters.

    Args:
        dataset (mogptk.dataset.DataSet): DataSet object of data for all channels. Only one channel allowed for SM.
        Q (int, optional): Number of components.
        name (str, optional): Name of the model.
        likelihood (gpflow.likelihoods, optional): Likelihood to use from GPFlow, if None a default exact inference Gaussian likelihood is used.
        variational (bool, optional): If True, use variational inference to approximate function values as Gaussian. If False it will use Monte Carlo Markov Chain.
        sparse (bool, optional): If True, will use sparse GP regression.
        like_params (dict, optional): Parameters to GPflow likelihood.

    Examples:

    >>> import numpy as np
    >>> t = np.linspace(0, 10, 100)
    >>> y = np.sin(0.5 * t)
    >>> import mogptk
    >>> data = mogptk.Data(t, y)
    >>> model = mogptk.SM([data], Q=1)
    >>> model.build()
    >>> model.train()
    >>> model.predict([np.linspace(1, 15, 150)])

    [1] A.G. Wilson and R.P. Adams, "Gaussian Process Kernels for Pattern Discovery and Extrapolation", International Conference on Machine Learning 30, 2013
    """
    def __init__(self, dataset, Q=1, name="SM"):
        model.__init__(self, name, dataset)
        self.Q = Q

        if self.dataset.get_output_dims() != 1:
            raise Exception("single output spectral mixture kernel can only have one output dimension in the data")

        kernel = MixtureKernel(SpectralKernel(self.dataset.get_input_dims()[0]), self.Q)
        self._build(kernel)

    def init_parameters(self, method='BNSE', noise=False):
        """
        Initialize parameters of kernel from data using different methods.

        Kernel parameters can be initialized using 3 heuristics using the train data:

        Arguments:
            method (str, optional): Method of estimation.
            noise (boolean, optional): Add noise of std.dev. equal to 1/10th of the estimated value.

        Methods:
            IPS:  Independant parameter sampling (from the PhD thesis of Andrew Wilson 2014)
                  takes the inverse of lengthscales drawn from truncated Gaussian N(0, max_dist^2),
                  the means drawn from Unif(0, 0.5 / minimum distance between two points),
                  and the mixture weights by taking the stdv of the y values divided by the
                  number of mixtures.
            LS:   Uses Lomb Scargle periodogram for estimating the PSD,
                  and using the first Q peaks as the means and mixture weights.
            BNSE: Uses the BNSE (Tobar 2018) to estimate the PSD 
                  and use the first Q peaks as the means and mixture weights.
        """

        if method.lower() not in ['ips', 'ls', 'bnse']:
            raise ValueError("valid methods of estimation are IPS, LS, and BNSE")

        if method.lower() == 'ips':
            input_dims = self.dataset[0].get_input_dims()
            nyquist = self.dataset[0].get_nyquist_estimation()
            x, y = self.dataset[0].get_train_data(transformed=True)
            x_range = np.max(x, axis=0) - np.min(x, axis=0)

            weights = [y.std()/self.Q] * self.Q
            means = nyquist * np.random.rand(self.Q, input_dims)
            variances = 1.0 / (np.abs(np.random.randn(self.Q, input_dims)) * x_range)

            for q in range(self.Q):
                self.model.kernel[q].weight.assign(weights[q])
                self.model.kernel[q].mean.assign(means[q,:])
                self.model.kernel[q].variance.assign(variances[q,:])
            return
        elif method.lower() == 'ls':
            amplitudes, means, variances = self.dataset[0].get_lombscargle_estimation(self.Q)
            if len(amplitudes) == 0:
                logger.warning('LS could not find peaks for SM')
                return
        elif method.lower() == 'bnse':
            amplitudes, means, variances = self.dataset[0].get_bnse_estimation(self.Q)
            if np.sum(amplitudes) == 0.0:
                logger.warning('BNSE could not find peaks for SM')
                return

        if noise:
            # TODO: what is this? check noise for all kernels
            pct = 1/30.0
            # noise proportional to the values
            noise_amp = np.random.multivariate_normal(
                mean=np.zeros(self.Q),
                cov=np.diag(amplitudes.mean(axis=1) * pct))
            # set value to a minimun value
            amplitudes = np.maximum(np.zeros_like(amplitudes) + 1e-6, amplitudes + noise_amp)

            noise_mean = np.random.multivariate_normal(
                mean=np.zeros(self.Q),
                cov=np.diag(means.mean(axis=1) * pct))
            means = np.maximum(np.zeros_like(means) + 1e-6, means + noise_mean)

            noise_var = np.random.multivariate_normal(
                mean=np.zeros(self.Q),
                cov=np.diag(variances.mean(axis=1) * pct))
            variances = np.maximum(np.zeros_like(variances) + 1e-6, variances + noise_var)

        # TODO: check weights for all kernels
        mixture_weights = amplitudes.mean(axis=1) / amplitudes.sum() * self.dataset[0].Y.transformed[self.dataset[0].mask].std() * 2

        for q in range(self.Q):
            self.model.kernel[q].weight.assign(mixture_weights[q])
            self.model.kernel[q].mean.assign(means[q,:])
            self.model.kernel[q].variance.assign(variances[q,:])

    def plot_psd(self, figsize=(10, 4), title='', log_scale=False):
        """
        Plot power spectral density of single output GP-SM.
        """
        #TODO: fix
        means = np.array([self.model.kernel[q].mean()*2.0*np.pi for q in range(self.Q)])
        weights = np.array([self.model.kernel[q].weight() for q in range(self.Q)])
        scales = np.array([self.model.kernel[q].variance() for q in range(self.Q)])
        
        # calculate bounds
        x_low = norm.ppf(0.001, loc=means, scale=scales).min()
        x_high = norm.ppf(0.99, loc=means, scale=scales).max()
        
        x = np.linspace(0, x_high + 1, 10000)

        psd = np.zeros_like(x)

        fig, axes = plt.subplots(1, 1, figsize=figsize)
        for q in range(self.Q):
            single_psd = weights[q] * norm.pdf(x, loc=means[q], scale=scales[q])
            axes.plot(x, single_psd, '--', lw=1.2, c='xkcd:strawberry', zorder=2)
            axes.axvline(means[q], ymin=0.001, ymax=0.1, lw=2, color='grey')
            psd = psd + single_psd
            
        # symmetrize PSD
        if psd[x<0].size != 0:
            psd = psd + np.r_[psd[x<0][::-1], np.zeros((x>=0).sum())]
            
        axes.plot(x, psd, lw=2.5, c='r', alpha=0.7, zorder=1)
        axes.set_xlim(0, x[-1] + 0.1)
        if log_scale:
            axes.set_yscale('log')
        axes.set_xlabel('Frequency [Hz]')
        axes.set_ylabel('PSD')
        axes.set_title(title)

        return fig, axes
