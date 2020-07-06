import numpy as np
from .model import model
from .dataset import DataSet
from .sm import _estimate_from_sm
# from .kernels import RestrictedMultiOutputSpectralMixture, Noise
from .kernels import MultiOutputSpectralMixture, RestrictedMultiOutputSpectralMixture_p, RestrictedMultiOutputSpectralMixture_u, Noise
from .plot import plot_spectrum
from .mosm import MOSM
import matplotlib as mpl
import matplotlib.pyplot as plt

import itertools
from scipy.special import comb


def _create_mask(M, Q_u, Q_p, Q_s):
    # total number of components
    Q = int(Q_u.sum() + Q_s + comb(M, 2) * Q_p)
    W = np.zeros((Q, M))
    modes = []

    # unique
    for i, q in enumerate(Q_u):
        if i==0:
            W[:q, i] = np.ones(q)
        else:
            aux = np.cumsum(Q_u)[i - 1]
            W[aux:aux+q, i] = np.ones(q)
        modes += ['u']*q
    # pairwise
    if Q_p > 0:
        pairs = list(itertools.combinations(range(M), 2))

        aux = Q_u.sum()
        for p in pairs:
            W[aux:aux+Q_p, p] = np.ones(Q_p)
            aux += Q_p
            modes.append('p')
    # shared
    if Q_s > 0:
        W[-Q_s:] = np.ones((Q_s, M))
        modes += ['s']*Q_s
    
    return W, Q, modes


class RMOSM(MOSM):
    """
    Restricted MOGP with Multi Output Spectral Mixture kernel.


    Variation of the MOSM kernel prosed in [1] where now each
    component can be either for: a single channel, a pair of channels
    or all channels. 
    When chosing only components for all channels original MOSM is
    recovered.
    The model contain the dataset and the associated gpflow model, 
    when the mogptk.Model is instanciated the gpflow model is built 
    using random parameters.

    Args:
        dataset (mogptk.dataset.DataSet): DataSet object of data for all channels.
        Q_u (list): Number of individual components per channel.
        Q_p (int): Number of pairwise components, is the number of components for each pair
            of channels.
        Q_s (int): Number of full shared components.
        name (str, optional): Name of the model.
        likelihood (gpflow.likelihoods, optional): Likelihood to use from GPFlow, if None a default exact inference Gaussian likelihood is used.
        variational (bool, optional): If True, use variational inference to approximate function values as Gaussian. If False it will use Monte Carlo Markov Chain.
        sparse (bool, optional): If True, will use sparse GP regression.
        like_params (dict, optional): Parameters to GPflow likelihood.

    Atributes:
        dataset: Constains the mogptk.DataSet associated.
        model: GPflow model.

    Examples:
    >>> import numpy as np
    >>> t = np.linspace(0, 10, 100)
    >>> y1 = np.sin(0.5 * t)
    >>> y2 = 2 * np.sin(0.2 * t)
    >>> import mogptk
    >>> data_list = []
    >>> mogptk.data_list.append(mogptk.Data(t, y1))
    >>> mogptk.data_list.append(mogptk.Data(t, y2))
    >>> model = mogptk.RMOSM(data_list, Q_u=[0, 2], Q_s=2, Q_p=1)
    >>> model.build()
    >>> model.train()
    >>> model.plot_prediction()

    [1] G. Parra and F. Tobar, "Spectral Mixture Kernels for Multi-Output Gaussian Processes", Advances in Neural Information Processing Systems 31, 2017
    """
    def __init__(
        self,
        dataset,
        Q_u=None,
        Q_p=1,
        Q_s=1,
        name="R-MOSM",
        likelihood=None,
        variational=False,
        sparse=False,
        inducing_variable=None,
        like_params={},
        magnitude_prior=None,
        **kwargs):

        model.__init__(self, name, dataset)

        M = self.dataset.get_output_dims()

        if Q_u is None:
            Q_u = np.zeros(M).astype(int)

        W, Q, modes = _create_mask(M, Q_u, Q_p, Q_s)

        self.Q = Q
        self.kernel_mask = W

        for q in range(Q):
            index = np.where(W[q, :] == 1)[0]
            if modes[q] == 's':
                kernel = MultiOutputSpectralMixture(
                self.dataset.get_input_dims()[0],
                self.dataset.get_output_dims(),
                magnitude_prior=magnitude_prior,
                )

            elif modes[q] == 'p':
                kernel = RestrictedMultiOutputSpectralMixture_p(
                self.dataset.get_input_dims()[0],
                self.dataset.get_output_dims(),
                magnitude_prior=magnitude_prior,
                channels=index,
                )

            else:
                kernel = RestrictedMultiOutputSpectralMixture_u(
                self.dataset.get_input_dims()[0],
                self.dataset.get_output_dims(),
                magnitude_prior=magnitude_prior,
                channels=index,
                )
            
            if q == 0:
                kernel_set = kernel
            else:
                kernel_set += kernel
        kernel_set += Noise(self.dataset.get_input_dims()[0], self.dataset.get_output_dims())
        self._build(kernel_set, likelihood, variational, sparse, like_params, inducing_variable, **kwargs)

    def init_parameters(self, method='BNSE', sm_method='BNSE', sm_opt='BFGS', sm_maxiter=3000, plot=False):
        """
        Initialize kernel parameters.

        The initialization can be done in two ways, the first by estimating the PSD via 
        BNSE (Tobar 2018) and then selecting the greater Q peaks in the estimated spectrum,
        the peaks position, magnitude and width initialize the mean, magnitude and variance
        of the kernel respectively.
        The second way is by fitting independent Gaussian process for each channel, each one
        with SM kernel, using the fitted parameters for initial values of the multioutput kernel.

        In all cases the noise is initialized with 1/30 of the variance 
        of each channel.

        Args:
            method (str, optional): Method of estimation, possible values are 'BNSE' and 'SM'.
            sm_method (str, optional): Method of estimating SM kernels. Only valid in 'SM' mode.
            sm_opt (str, optional): Optimization method for SM kernels. Only valid in 'SM' mode.
            sm_maxiter (str, optional): Maximum iteration for SM kernels. Only valid in 'SM' mode.
            plot (bool, optional): Show the PSD of the kernel after fitting SM kernels. Only valid in 'SM' mode.
        """

        n_channels = self.dataset.get_output_dims()

        W = self.kernel_mask

        if method in ['BNSE', 'LS']:
            if method == 'BNSE':
                amplitudes, means, variances = self.dataset.get_bnse_estimation(self.Q)
            else:
                amplitudes, means, variances = self.dataset.get_lombscargle_estimation(self.Q)
            if len(amplitudes) == 0:
                logger.warning('BNSE could not find peaks for MOSM')
                return

            magnitude = np.zeros((n_channels, self.Q))
            for q in range(self.Q):
                mean = np.empty((self.dataset.get_input_dims()[0], n_channels))
                variance = np.empty((self.dataset.get_input_dims()[0], n_channels))
                for channel in range(n_channels):
                    magnitude[channel, q] = amplitudes[channel][:,q].mean()
                    mean[:, channel] = means[channel][:,q] * 2 * np.pi
                    variance[:, channel] = variances[channel][:,q] * 2

                index = W[q, :].astype(bool)
                self.set_parameter(q, 'mean', mean[:, index])
                self.set_parameter(q, 'variance', variance[:, index])
            
            # normalize proportional to channels variances
            magnitude = magnitude * W.T
            for channel, data in enumerate(self.dataset):
                magnitude[channel, :] = np.sqrt(magnitude[channel, :] / magnitude[channel, :].sum() * data.Y.transformed[data.mask].var()) * 2

            for q in range(self.Q):
                index = W[q, :].astype(bool)
                self.set_parameter(q, 'magnitude', magnitude[index, q])

        elif method == 'SM':
            params = _estimate_from_sm(self.dataset, self.Q, method=sm_method, optimizer=sm_opt, maxiter=sm_maxiter, plot=plot)

            magnitude = np.zeros((n_channels, self.Q))
            for q in range(self.Q):
                index = W[q, :].astype(bool)
                magnitude[:, q] = params[q]['weight'].mean(axis=0)
                self.set_parameter(q, 'mean', params[q]['mean'][:, index])
                self.set_parameter(q, 'variance', params[q]['scale'][:, index] * 2)

            # normalize proportional to channels variances
            magnitude = magnitude * W.T
            for channel, data in enumerate(self.dataset):
                if magnitude[channel, :].sum()==0:
                    raise Exception("Sum of magnitudes equal to zero")

                magnitude[channel, :] = np.sqrt(magnitude[channel, :] / magnitude[channel, :].sum() * data.Y.transformed[data.mask].var()) * 2
            for q in range(self.Q):
                self.set_parameter(q, 'magnitude', magnitude[index, q])
        else:
            raise Exception("possible methods of estimation are either 'BNSE' or 'SM'")

        noise = np.empty((n_channels))
        for channel, data in enumerate(self.dataset):
            noise[channel] = (data.Y.transformed[data.mask]).var() / 30
        self.set_parameter(self.Q, 'noise', noise)
