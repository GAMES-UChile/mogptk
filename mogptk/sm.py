import numpy as np
from .model import model
from .kernels import SpectralMixture, sm_init, Noise
from .plot import plot_spectrum
from scipy.stats import norm
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger('mogptk')

def _estimate_from_sm(dataset, Q, method='BNSE', optimizer='BFGS', maxiter=2000, plot=False, fix_means=False):
    """
    Estimate kernel param with single ouput GP-SM

    Args:
        dataset (mogptk.DataSet): DataSet object of data with one channel.
        Q (int): Number of components.
        estimate (str): Method to estimate, 'BNSE', 'LS' or 'AGW'
        method (str): Optimization method, either 'Adam' or any
            scipy optimizer.
        maxiter (int): Maximum number of iteration.
        plot (bool): If true plot the kernel PSD of each channel.
        fix_means(bool): Fix spectral means to zero in trainning.

    Returns: 
        params[q][name][output dim][input dim]
    """
    input_dims = dataset.get_input_dims()[0]
    output_dims = dataset.get_output_dims()

    params = []
    for q in range(Q):
        params.append({
            'weight': np.empty((input_dims, output_dims)),
            'mean': np.empty((input_dims, output_dims)),
            'scale': np.empty((input_dims, output_dims)),
        })

    for channel in range(output_dims):
        for i in range(input_dims):  # TODO one SM per channel
            sm = SM(dataset[channel], Q)
            sm.init_parameters(method)

            if fix_means:
                sm.set_parameter(0, 'mixture_means', np.zeros((Q, input_dims)))
                sm.set_parameter(0, 'mixture_scales', sm.get_parameter(0, 'mixture_scales') * 50)
                sm.fix_parameters(0, 'mixture_means')

            sm.train(method=optimizer, maxiter=maxiter)

            if plot:
                nyquist = dataset[channel].get_nyquist_estimation()
                means = sm.get_parameter(0, 'mixture_means')
                weights = sm.get_parameter(0, 'mixture_weights')
                scales = sm.get_parameter(0, 'mixture_scales').T
                plot_spectrum(means, scales, weights=weights, nyquist=nyquist, title=dataset[channel].name)

            for q in range(Q):
                params[q]['weight'][i,channel] = sm.get_parameter(0, 'mixture_weights')[q]
                params[q]['mean'][i,channel] = sm.get_parameter(0, 'mixture_means')[q,:] * np.pi * 2
                params[q]['scale'][i,channel] = sm.get_parameter(0, 'mixture_scales')[:,q]

    return params

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
    def __init__(self, dataset, Q=1, name="SM", likelihood=None, variational=False, sparse=False, like_params={}):
        model.__init__(self, name, dataset)
        self.Q = Q

        if self.dataset.get_output_dims() != 1:
            raise Exception("single output spectral mixture kernel can only have one output dimension in the data")

        kernel = SpectralMixture(
            self.dataset.get_input_dims()[0],
            self.Q,
        )
        self._build(kernel, likelihood, variational, sparse, like_params)

    def init_parameters(self, method='BNSE'):
        """
        Initialize parameters of kernel from data using different methods.

        Kernel parameters can be initialized using 3 heuristics using the train data:

        'AGW': (Taken from phd thesis from Andrew wilson 2014) is taking the inverse
            of lengthscales drawn from truncated Gaussian N(0, max_dist^2), the
            means drawn from Unif(0, 0.5 / minimum distance between two points),
            and the mixture weights by taking the stdv of the y values divided by the
            number of mixtures.
        'LS'_ is using Lomb Scargle periodogram for estimating the PSD,
            and using the first Q peaks as the means and mixture weights.
        'BNSE': third is using the BNSE (Tobar 2018) to estimate the PSD 
            and use the first Q peaks as the means and mixture weights.

        In all cases the noise is initialized with 1/30 of the variance of each channel.

        *** Only for single input dimension for each channel.
        """

        if method not in ['AGW', 'LS', 'BNSE']:
            raise Exception("possible methods are 'AGW', 'LS' and 'BNSE' (see documentation).")

        if method == 'AGW':
            x, y = self.dataset[0].get_train_data()
            weights, means, scales = sm_init(x, y, self.Q)
            self.set_parameter(0, 'mixture_weights', weights)
            self.set_parameter(0, 'mixture_means', np.array(means))
            self.set_parameter(0, 'mixture_scales', np.array(scales.T))

        elif method == 'LS':
            amplitudes, means, variances = self.dataset[0].get_lombscargle_estimation(self.Q)
            if len(amplitudes) == 0:
                logger.warning('LS could not find peaks for SM')
                return
            
            mixture_weights = amplitudes.mean(axis=0) / amplitudes.sum() * self.dataset[0].Y[self.dataset[0].mask].var() * 2

            self.set_parameter(0, 'mixture_weights', mixture_weights)
            self.set_parameter(0, 'mixture_means', means.T)
            self.set_parameter(0, 'mixture_scales', variances * 2.0)

        elif method == 'BNSE':
            amplitudes, means, variances = self.dataset[0].get_bnse_estimation(self.Q)
            if np.sum(amplitudes) == 0.0:
                logger.warning('BNSE could not find peaks for SM')
                return

            mixture_weights = amplitudes.mean(axis=0) / amplitudes.sum() * self.dataset[0].Y[self.dataset[0].mask].var() * 2

            self.set_parameter(0, 'mixture_weights', mixture_weights)
            self.set_parameter(0, 'mixture_means', means.T)
            self.set_parameter(0, 'mixture_scales', variances * 2.0)

    def plot_psd(self, figsize=(10, 4), title='', log_scale=False):
        """
        Plot power spectral density of single output GP-SM.
        """
        means = self.get_parameter(0, 'mixture_means') * 2.0 * np.pi
        weights = self.get_parameter(0, 'mixture_weights')
        scales = self.get_parameter(0, 'mixture_scales').T
        
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
