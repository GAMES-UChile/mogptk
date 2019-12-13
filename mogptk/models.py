import logging
import dill
from .plot import plot_spectrum
from .data import Data
from .model import model
from .kernels import SpectralMixture, sm_init, MultiOutputSpectralMixture, SpectralMixtureLMC, ConvolutionalGaussian, CrossSpectralMixture, Noise
from .kernels.conv_old import ConvolutionalGaussianOLD
import numpy as np
import gpflow
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import norm

def LoadModel(filename):
    """
    Load a model from a given file that was previously saved (see the `model.save()` function).
    """
    if not filename.endswith(".mogptk"):
        filename += ".mogptk"

    graph = tf.Graph()
    session = tf.Session(graph=graph)
    with graph.as_default():
        with session.as_default():
            model = gpflow.saver.Saver().load(filename)
            model_type = model.mogptk_type
            name = model.mogptk_name
            data = dill.loads(eval(model.mogptk_data))
            Q = model.mogptk_Q
            params = model.mogptk_params

    if model_type == 'SM':
        m = SM(data, Q, name)
    elif model_type == 'MOSM':
        m = MOSM(data, Q, name)
    elif model_type == 'CG':
        m = CG(data, Q, name)
    elif model_type == 'CSM':
        m = CSM(data, Q, name)
    elif model_type == 'SM_LMC':
        m = SM_LMC(data, Q, name)
    else:
        raise Exception("unknown model type '%s'" % (model_type))

    m.model = model
    m.params = params
    m.graph = graph
    m.session = session
    m.build() # TODO: should not be necessary
    return m

def _transform_data_mogp(x, y):
    chan = []
    for channel in range(len(x)):
        chan.append(channel * np.ones(len(x[channel])))
    chan = np.concatenate(chan).reshape(-1, 1)
    
    x = np.concatenate(x)
    x = np.concatenate((chan, x), axis=1)
    if y == None:
        return x

    y = np.concatenate(y).reshape(-1, 1)
    return x, y


def _estimate_noise_var(data):
    """
    Estimate noise variance with a tenth of the channel variance
    """

    output_dims = len(data)

    noise_var = np.zeros(output_dims)

    for i, channel in enumerate(data):
        noise_var[i] = (channel.Y).var() / 30

    return noise_var


def _estimate_from_sm(data, Q, init='BNSE', method='BFGS', maxiter=2000, plot=False, fix_means=False):
    """
    Estimate kernel param with single ouput GP-SM

    Args:
        data (obj; mogptk.Data): Data class instance.

        Q (int): Number of components.
        init (str): Method to initialize, 'BNSE', 'LS' or 'Random'
        method (str): Optimization method, either 'Adam' or any
            scipy optimizer.
        maxiter (int): Maximum number of iteration.
        plot (bool): If true plot the kernel PSD of each channel.
        fix_means(bool): Fix spectral means to zero in trainning.

    returns: 
        params[q][name][output dim][input dim]
    """
    input_dims = data[0].get_input_dims()
    output_dims = len(data)

    params = []
    for q in range(Q):
        params.append({
            'weight': np.empty((input_dims, output_dims)),
            'mean': np.empty((input_dims, output_dims)),
            'scale': np.empty((input_dims, output_dims)),
        })

    for channel in range(output_dims):
        for i in range(input_dims):  # TODO one SM per channel
            sm = SM(data[channel], Q)
            sm.init_params(init)

            if fix_means:
                for q in range(sm.Q):
                    sm.params[q]['mixture_means'] = np.zeros(input_dims)
                    sm.params[q]['mixture_scales'] *= 100

            sm.build()

            if fix_means:
                sm.fix_param('mixture_means')

            sm.train(method=method, maxiter=maxiter, tol=1e-50)

            if plot:
                nyquist = data[channel].get_nyquist_estimation()
                means = sm._get_param_across('mixture_means')
                weights = sm._get_param_across('mixture_weights')
                scales = sm._get_param_across('mixture_scales')
                plot_spectrum(means, scales, weights=weights, nyquist=nyquist, title=data[channel].name)

            for q in range(Q):
                params[q]['weight'][i,channel] = sm.params[q]['mixture_weights']
                params[q]['mean'][i,channel] = sm.params[q]['mixture_means'] * np.pi * 2
                params[q]['scale'][i,channel] = sm.params[q]['mixture_scales']

    return params

class SM(model):
    """
    Single output GP with Spectral mixture kernel.

    A Gaussian process regression using Spectral mixture kernel [1]

    Args:
        data (list of mogptk.Data): List of channels, each element should be
            a mogptk.Data object.
        Q (int): Number of components.
        name (str): Name of the model.

    Atributes:
        data (list of mogptk.Data): List of channels, each element is 
            a mogptk.Data object.
        params (dict): Dictionary of lenght Q with kernel parameters,
            the elements are dicts with parameters of each components.
    ----------
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

    References:
    [1] A. Wilson and R. Adams, “Gaussian process kernels for pattern discovery and extrapolation,”
    in International Conference on Machine Learning, 2013.
    """
    def __init__(self, data, Q=1, name="SM"):
        model.__init__(self, name, data, Q)

        input_dims = self.get_input_dims()
        output_dims = self.get_output_dims()
        if output_dims != 1:
            raise Exception("Single output Spectral Mixture kernel can only take one output dimension in the data")

        weights = np.abs(np.random.standard_normal((Q)))
        means = np.abs(np.random.standard_normal((Q, input_dims)))
        scales = np.abs(np.random.standard_normal((Q, input_dims)))

        for q in range(Q):
            self.params.append({
                'mixture_weights': weights[q],
                'mixture_means': np.array(means[q]),
                'mixture_scales': np.array(scales[q]),
            })

    def init_params(self, method='BNSE'):
        """
        Initialize parameters of kernel from data using different methods.

        Kernel parameters can be initialized using 3 heuristics using the train data:

        'random': (Taken from phd thesis from Andrew wilson 2014) is taking the inverse
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

        if method not in ['random', 'LS', 'BNSE']:
            raise Exception("Posible methods are 'random', 'LS' and 'BNSE' (see documentation).")

        if method=='random':
            x, y = self.data[0].get_obs()
            weights, means, scales = sm_init(x, y, self.Q)
            for q in range(self.Q):
                self.params[q]['mixture_weights'] = weights[q]
                self.params[q]['mixture_means'] = np.array(means[q])
                self.params[q]['mixture_scales'] = np.array(scales[q])

        elif method=='LS':
            amplitudes, means, variances = self.data[0].get_ls_estimation(self.Q)
            if len(amplitudes) == 0:
                logging.warning('BNSE could not find peaks for SM')
                return

            for q in range(self.Q):
                self.params[q]['mixture_weights'] = amplitudes[:, q].mean() / amplitudes.mean()
                self.params[q]['mixture_means'] = means.T[q]
                self.params[q]['mixture_scales'] = variances.T[q] * 2

        elif method=='BNSE':
            amplitudes, means, variances = self.data[0].get_bnse_estimation(self.Q)
            if np.sum(amplitudes) == 0.0:
                logging.warning('BNSE could not find peaks for SM')
                return

            for q in range(self.Q):
                self.params[q]['mixture_weights'] = amplitudes[:, q].mean() / amplitudes.mean()
                self.params[q]['mixture_means'] = means.T[q]
                self.params[q]['mixture_scales'] = variances.T[q] * 2

    def _transform_data(self, x, y=None):
        if y == None:
            return x[0]
        return x[0], np.expand_dims(y[0], 1)

    def _kernel(self):
        weights = np.array([self.params[q]['mixture_weights'] for q in range(self.Q)])
        means = np.array([self.params[q]['mixture_means'] for q in range(self.Q)])
        scales = np.array([self.params[q]['mixture_scales'] for q in range(self.Q)]).T
        return SpectralMixture(
            self.Q,
            weights,
            means,
            scales,
            self.get_input_dims(),
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


    def plot_psd(self, figsize=(10, 4), title='', log_scale=False):
        """
        Plot power spectral density of 
        single output GP-SM
        """
        means = self._get_param_across('mixture_means').reshape(-1)
        weights = self._get_param_across('mixture_weights').reshape(-1)
        scales = self._get_param_across('mixture_scales').reshape(-1)
        
        # calculate bounds
        x_low = norm.ppf(0.001, loc=means, scale=scales).min()
        x_high = norm.ppf(0.99, loc=means, scale=scales).max()
        
        x = np.linspace(x_low, x_high + 1, 1000)
        psd = np.zeros_like(x)

        f, ax = plt.subplots(1, 1, figsize=figsize)
        # fig, axes = plt.subplots(1, 1, figsize=(20, 5)
        
        for q in range(self.Q):
            single_psd = weights[q] * norm.pdf(x, loc=means[q], scale=scales[q])
            ax.plot(x, single_psd, '--', lw=1.2, c='xkcd:strawberry', zorder=2)
            ax.axvline(means[q], ymin=0.001, ymax=0.1, lw=2, color='grey')
            psd = psd + single_psd
            
        # symmetrize PSD
        if psd[x<0].size != 0:
            psd = psd + np.r_[psd[x<0][::-1], np.zeros((x>=0).sum())]
            
        ax.plot(x, psd, lw=2.5, c='r', alpha=0.7, zorder=1)
        ax.set_xlim(0, x[-1] + 0.1)
        if log_scale:
            ax.set_yscale('log')
        ax.set_xlabel(r'$\omega$')
        ax.set_ylabel('PSD')
        ax.set_title(title)

class MOSM(model):
    """
    MOGP with Multi Output Spectral Mixture kernel, as proposed in [1].

    Args:
        data (list of mogptk.Data): List of channels, each element should be
            a mogptk.Data object.
        Q (int): Number of components.
        name (str): Name of the model.
        prior (gpflow.prior): Prior on the magnitudes.

    Atributes:
        data (list of mogptk.Data): List of channels, each element is 
            a mogptk.Data object.
        Q (int): Number of components.
        params (dict): Dictionary of lenght Q + 1 with kernel parameters,
            the first Q elements are dicts with parameters of each componentes,
            the last element is the noise variance.
        prior (gpflow.prior): Prior on the magnitudes.

    ----------
    Examples:
    >>> import numpy as np
    >>> t = np.linspace(0, 10, 100)
    >>> y1 = np.sin(0.5 * t)
    >>> y2 = 2 * np.sin(0.2 * t)
    >>> import mogptk
    >>> data_list = []
    >>> mogptk.data_list.append(mogptk.Data(t, y1))
    >>> mogptk.data_list.append(mogptk.Data(t, y2))
    >>> model = mogptk.MOSM(data_list, Q=2)
    >>> model.build()
    >>> model.train()
    >>> model.plot_prediction()

    References:
    [1] G. Parra and F. Tobar, “Spectral mixture kernels for multioutput Gaussian processes,”
    Advances in Neural Information Processing Systems, 2017.
    """
    def __init__(self, data, Q=1, name="MOSM", prior=None):
        model.__init__(self, name, data, Q)

        input_dims = self.get_input_dims()
        output_dims = self.get_output_dims()
        for _ in range(Q):
            self.params.append({
                "magnitude": np.random.standard_normal((output_dims)),
                "mean": np.random.standard_normal((input_dims, output_dims)),
                "variance": np.random.random((input_dims, output_dims)),
                "delay": np.zeros((input_dims, output_dims)),
                "phase": np.zeros((output_dims)),
            })
        self.params.append({
            "noise": np.random.random((output_dims)),
        })
        self.prior = prior

    def init_means(self):
        """
        (DEPRECATED)

        Initialize spectral means using BNSE[1]

        """
        # peaks, _ = self.data.get_bnse_estimation(self.Q)
        # peaks = np.array(peaks)
        # for q in range(self.Q):
        #    self.params[q]["mean"] = peaks[0].T[q].reshape(-1, 1)
        #    for channel in range(1,self.data.get_output_dims()):
        #        self.params[q]["mean"] = np.append(self.params[q]["mean"], peaks[channel].T[q].reshape(-1, 1))

        for channel in range(self.get_output_dims()):
            amplitudes, means, variances = self.data[channel].get_bnse_estimation(self.Q)

            for q in range(self.Q):
                self.params[q]["mean"][:, channel] = means[:, q] * 2 * np.pi
                self.params[q]["magnitude"][channel] = amplitudes[:, q].mean()
                self.params[q]["variance"][:, channel] = variances[:, q] * 2

        # normalize across channels
        for q in range(self.Q):
            self.params[q]["magnitude"] = np.sqrt(self.params[q]["magnitude"] / self.params[q]["magnitude"].mean())


    def init_params(self, mode='BNSE', sm_init='BNSE', sm_method='BFGS', sm_maxiter=3000, plot=False):
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
            mode (str): Method of initializing, possible values are 'BNSE' and SM.
            sm_init (str): Method of initializing SM kernels. Only valid in 'SM' mode.
            sm_method (str): Optimization method for SM kernels. Only valid in 'SM' mode.
            sm_maxiter (str): Maximum iteration for SM kernels. Only valid in 'SM' mode.
            plt (bool): Show the PSD of the kernel after fitting SM kernels.
                Only valid in 'SM' mode. Default to false.
        """

        if mode=='BNSE':
            for channel in range(self.get_output_dims()):
                amplitudes, means, variances = self.data[channel].get_bnse_estimation(self.Q)

                for q in range(self.Q):
                    self.params[q]["mean"][:, channel] = means[:, q] * 2 * np.pi
                    self.params[q]["magnitude"][channel] = amplitudes[:, q].mean()
                    self.params[q]["variance"][:, channel] = variances[:, q] * 2

            # normalize across channels
            for q in range(self.Q):
                self.params[q]["magnitude"] = np.sqrt(self.params[q]["magnitude"] / self.params[q]["magnitude"].mean())

        elif mode=='SM':
            params = _estimate_from_sm(
                self.data,
                self.Q,
                init=sm_init,
                method=sm_method,
                maxiter=sm_maxiter,
                plot=plot)

            for q in range(self.Q):
                self.params[q]["magnitude"] = params[q]['weight'].mean(axis=0)
                self.params[q]["mean"] = params[q]['mean']
                self.params[q]["variance"] = params[q]['scale'] * 2

                self.params[q]["magnitude"] = np.sqrt(self.params[q]["magnitude"] / self.params[q]["magnitude"].mean())
        else:
            raise Exception("possible modes are either 'BNSE' or 'SM'")

        # noise init
        self.params[self.Q]['noise'] = _estimate_noise_var(self.data)

    def plot(self):
        names = [channel.name for channel in self.data]
        nyquist = [channel.get_nyquist_estimation() for channel in self.data]
        means = self._get_param_across('mean')
        weights = self._get_param_across('magnitude')**2
        scales = self._get_param_across('variance')
        plot_spectrum(means, scales, weights=weights, nyquist=nyquist, titles=names)

    def plot_psd(self, figsize=(20, 14), title=''):
        """
        Plot power spectral density and power cross spectral density.

        Note: Implemented only for 1 input dimension.
        """

        cross_params = self.get_cross_params()

        m = self.get_output_dims()

        f, axarr = plt.subplots(m, m, sharex=False, figsize=figsize)

        for i in range(m):
            for j in range(i+1):
                self._plot_power_cross_spectral_density(
                    axarr[i, j],
                    cross_params,
                    channels=(i, j))

        plt.tight_layout()
        return f, axarr

    def _plot_power_cross_spectral_density(self, ax, params, channels=(0, 0)):
        """
        Plot power cross spectral density given axis.

        Args:
            ax (matplotlib.axis): Axis to plot to.
            params(dict): Kernel parameters.
            channels (tuple of ints): Channels to use.
        """
        i = channels[0]
        j = channels[1]

        mean = params['mean'][i, j, 0, :]
        cov = params['covariance'][i, j, 0, :]
        delay = params['delay'][i, j, 0, :]
        magn = params['magnitude'][i, j, :]
        phase = params['phase'][i, j, :]

        
        w_high = (mean + 1* np.sqrt(cov)).max()

        w = np.linspace(-w_high, w_high, 1000)

        # power spectral density
        if i==j:
            psd_total = np.zeros(len(w))
            for q in range(self.Q):
                psd_q = np.exp(-0.5 * (w - mean[q])**2 / cov[q])
                psd_q += np.exp(-0.5 * (w + mean[q])**2 / cov[q])
                psd_q *= magn[q] * 0.5

                ax.plot(w, psd_q, '--r', lw=0.5)

                psd_total += psd_q
            ax.plot(w, psd_total, 'r', lw=2.1, alpha=0.7)
        # power cross spectral density
        else:
            psd_total = np.zeros(len(w)) + 0.j
            for q in range(self.Q):
                psd_q = np.exp(-0.5 * (w - mean[q])**2 / cov[q] + 1.j * (w * delay[q] + phase[q]))
                psd_q += np.exp(-0.5 * (w + mean[q])**2 / cov[q] + 1.j * (w * delay[q] + phase[q]))
                psd_q *= magn[q] * 0.5

                ax.plot(w, np.real(psd_q), '--b', lw=0.5)
                ax.plot(w, np.imag(psd_q), '--g', lw=0.5)
            
                psd_total += psd_q
            ax.plot(w, np.real(psd_total), 'b', lw=1.2, alpha=0.7)
            ax.plot(w, np.imag(psd_total), 'g', lw=1.2, alpha=0.7)
        ax.set_yticks([])
        return

    def plot_correlations(self, figsize=None):
        """
        Plot correlation coeficient matrix.

        This is done evaluating the kernel at K_ij(0, 0)
        for al channels.
        """

        m = self.get_output_dims()

        cross_params = self.get_cross_params()
        mean = cross_params['mean'][:, :, 0, :]
        cov = cross_params['covariance'][:, :, 0, :]
        delay = cross_params['delay'][:, :, 0, :]
        magn = cross_params['magnitude'][:, :, :]
        phase = cross_params['phase'][:, :, :]

        corr_coef_matrix = np.zeros((m, m))

        alpha = magn / np.sqrt(2 * np.pi * cov)

        np.fill_diagonal(corr_coef_matrix, np.diagonal(alpha.sum(2)))

        for i in range(m):
            for j in range(m):
                if i!=j:
                    corr = np.exp(-0.5 *  delay[i, j, :]**2 * cov[i, j, :]) 
                    corr *= np.cos(delay[i, j, :] * mean[i, j, :] + phase[i, j, :])
                    corr_coef_matrix[i, j] = (alpha[i, j, :] * corr).sum()
                norm = np.sqrt(np.diagonal(alpha.sum(2))[i]) * np.sqrt(np.diagonal(alpha.sum(2))[j])
                corr_coef_matrix[i, j] /= norm

        fig, ax = plt.subplots()
        color_range = np.abs(corr_coef_matrix).max()
        norm = mpl.colors.Normalize(vmin=-color_range, vmax=color_range)
        im = ax.matshow(corr_coef_matrix, cmap='coolwarm', norm=norm)
        fig.colorbar(im)
        for (i, j), z in np.ndenumerate(corr_coef_matrix):
            ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5, edgecolor='0.9'),
                fontsize=figsize)
        return fig, ax, corr_coef_matrix

    def info(self):
        for channel in range(self.get_output_dims()):
            for q in range(self.Q):
                mean = self.params[q]["mean"][:,channel]
                var = self.params[q]["variance"][:,channel]
                if np.linalg.norm(mean) < np.linalg.norm(var):
                    print("‣ MOSM approaches RBF kernel for Q=%d in channel='%s'" % (q, self.data[channel].name))

    def get_cross_params(self):
        """
        Obtain cross parameters from MOSM

        Returns:
            cross_params(dict): Dictionary with the cross parameters, 'covariance', 'mean',
            'magnitude', 'delay' and 'phase'. Each one a output_dim x output_dim x input_dim x Q
            array with the cross parameters, with the exception of 'magnitude' and 'phase' where 
            the cross parameters are a output_dim x output_dim x Q array.
            NOTE: this assumes the same input dimension for all channels.
        """
        m = self.get_output_dims()
        d = self.get_input_dims()
        Q = self.Q

        cross_params = {}

        cross_params['covariance'] = np.zeros((m, m, d, Q))
        cross_params['mean'] = np.zeros((m, m, d, Q))
        cross_params['magnitude'] = np.zeros((m, m, Q))
        cross_params['delay'] = np.zeros((m, m, d, Q))
        cross_params['phase'] = np.zeros((m, m, Q))

        for q in range(Q):
            for i in range(m):
                for j in range(m):
                    var_i = self.params[q]['variance'][:, i]
                    var_j = self.params[q]['variance'][:, j]
                    mu_i = self.params[q]['mean'][:, i]
                    mu_j = self.params[q]['mean'][:, j]
                    w_i = self.params[q]['magnitude'][i]
                    w_j = self.params[q]['magnitude'][j]
                    sv = var_i + var_j

                    # cross covariance
                    cross_params['covariance'][i, j, :, q] = 2 * (var_i * var_j) / sv
                    # cross mean
                    cross_mean_num = var_i.dot(mu_j) + var_j.dot(mu_i)
                    cross_params['mean'][i, j, :, q] = cross_mean_num / sv
                    # cross magnitude
                    exp_term = -1/4 * ((mu_i - mu_j)**2 / sv).sum()
                    cross_params['magnitude'][i, j, q] = w_i * w_j * np.exp(exp_term)
            # cross phase
            phase_q = self.params[q]['phase']
            cross_params['phase'][:, :, q] = np.subtract.outer(phase_q, phase_q)
            for n in range(d):
                # cross delay        
                delay_n_q = self.params[q]['delay'][n, :]
                cross_params['delay'][:, :, n, q] = np.subtract.outer(delay_n_q, delay_n_q)

        return cross_params

    def _transform_data(self, x, y=None):
        return _transform_data_mogp(x, y)

    def _kernel(self):
        for q in range(self.Q):
            kernel = MultiOutputSpectralMixture(
                self.get_input_dims(),
                self.get_output_dims(),
                self.params[q]["magnitude"],
                self.params[q]["mean"],
                self.params[q]["variance"],
                self.params[q]["delay"],
                self.params[q]["phase"],
                magnitude_prior=self.prior,
            )
            if q == 0:
                kernel_set = kernel
            else:
                kernel_set += kernel
        kernel_set += Noise(self.get_input_dims(), self.get_output_dims(), self.params[len(self.params)-1]["noise"])
        return kernel_set

class CSM(model):
    """
    Cross Spectral Mixture kernel [1] with Q components and Rq latent functions.

    Args:
        data (list of mogptk.Data): List of channels, each element should be
            a mogptk.Data object.
        Q (int): Number of components.
        Rq (int): Sub components por components.
        name (str): Name of the model.

    Atributes:
        data (list of mogptk.Data): List of channels, each element is 
            a mogptk.Data object.
        Q (int): Number of components.
        params (dict): Dictionary of lenght Q + 1 with kernel parameters,
            the first Q elements are dicts with parameters of each componentes,
            the last element is the noise variance.

    ----------
    Examples:
    >>> import numpy as np
    >>> t = np.linspace(0, 10, 100)
    >>> y1 = np.sin(0.5 * t)
    >>> y2 = 2 * np.sin(0.2 * t)
    >>> import mogptk
    >>> data_list = []
    >>> mogptk.data_list.append(mogptk.Data(t, y1))
    >>> mogptk.data_list.append(mogptk.Data(t, y2))
    >>> model = mogptk.CSM(data_list, Q=2)
    >>> model.build()
    >>> model.train()
    >>> model.plot_prediction()

    References:
    [1] Ulrich et al, “GP kernels for cross-spectrum analysis”,
    Advances in neural information processing systems, 2015.
    """
    def __init__(self, data, Q=1, Rq=1, name="CSM"):
        model.__init__(self, name, data, Q)

        if Rq != 1:
            raise Exception("Rq != 1 is not (yet) supported") # TODO: support
        self.Rq = Rq
        
        input_dims = self.get_input_dims()
        output_dims = self.get_output_dims()
        for _ in range(Q):
            self.params.append({
                "constant": np.random.random((Rq, output_dims)),
                "mean": np.random.random((input_dims)),
                "variance": np.random.random((input_dims)),
                "phase": np.zeros((Rq, output_dims)),
            })
        self.params.append({
            "noise": np.random.random((output_dims)),
        })
    
    def init_params(self, method='BNSE', sm_init='BNSE', sm_method='BFGS', sm_maxiter=3000, plot=False):
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
            mode (str): Method of initializing, possible values are 'BNSE' and SM.
            sm_init (str): Method of initializing SM kernels. Only valid in 'SM' mode.
            sm_method (str): Optimization method for SM kernels. Only valid in 'SM' mode.
            sm_maxiter (str): Maximum iteration for SM kernels. Only valid in 'SM' mode.
            plt (bool): Show the PSD of the kernel after fitting SM kernels.
                Only valid in 'SM' mode. Default to false.
        """

        # data = self.data.copy()
        # data.normalize()
        if method == 'BNSE':

            means = np.zeros((self.get_input_dims(), self.Q))
            variances = np.zeros((self.get_input_dims(), self.Q))

            for channel in range(self.get_output_dims()):
                single_amp, single_mean, single_var = self.data[channel].get_bnse_estimation(self.Q)
                means += single_mean
                variances += single_var

                for q in range(self.Q):
                    self.params[q]["constant"][:, channel] = single_amp[:, q].mean()

            # get mean across channels
            means *= 1 / self.get_output_dims()
            variances *= 1 / self.get_output_dims()

            for q in range(self.Q):
                self.params[q]["mean"] = means[:, q] * 2 * np.pi
                self.params[q]["variance"] = variances[:, q] * 5

                # normalize across channels
                self.params[q]["constant"] = self.params[q]["constant"] / self.params[q]["constant"].max()
        elif method == 'SM':
            # all_params = _estimate_from_sm(self.data, self.Q, init=sm_init, method=sm_method, maxiter=sm_maxiter, plot=plot)
            
            # input_dims = self.get_input_dims()
            # output_dims = self.get_output_dims()
            # params = {
            #     'weight': np.zeros((self.Q*output_dims)),
            #     'mean': np.zeros((self.Q*output_dims, input_dims)),
            #     'scale': np.zeros((self.Q*output_dims, input_dims)),
            # }
            # for channel in range(output_dims):
            #     for q in range(self.Q):
            #         weight = np.average(all_params[q]['weight'][:,channel])
            #         if weight != 0.0:
            #             weight /= np.sum(all_params[q]['weight'])
            #         mean = all_params[q]['mean'][:,channel].reshape(1, -1)
            #         scale = all_params[q]['scale'][:,channel].reshape(1, -1)
            #         params['weight'][channel*q+q] = weight
            #         params['mean'][channel*q+q,:] = mean
            #         params['scale'][channel*q+q,:] = scale

            # indices = np.argsort(params['weight'])[::-1]
            # for q in range(self.Q):
            #     if q < len(indices):
            #         i = indices[q]
            #         self.params[q]['mean'] = params['mean'][i]
            #         self.params[q]['variance'] = params['scale'][i] * 2
            params = _estimate_from_sm(self.data, self.Q, init=sm_init, method=sm_method, maxiter=sm_maxiter, plot=plot)
            for q in range(self.Q):
                self.params[q]["constant"] = params[q]['weight'].mean(axis=0).reshape(self.Rq, -1)
                self.params[q]["mean"] = params[q]['mean'].mean(axis=1)
                self.params[q]["variance"] = params[q]['scale'].mean(axis=1) * 2
        else:
            raise Exception("possible modes are either 'BNSE' or 'SM'")

        # noise init
        self.params[self.Q]['noise'] = _estimate_noise_var(self.data)

    def _transform_data(self, x, y=None):
        return _transform_data_mogp(x, y)

    def _kernel(self):
        for q in range(self.Q):
            kernel = CrossSpectralMixture(
                self.get_input_dims(),
                self.get_output_dims(),
                self.Rq,
                self.params[q]["constant"],
                self.params[q]["mean"],
                self.params[q]["variance"],
                self.params[q]["phase"],
            )
            if q == 0:
                kernel_set = kernel
            else:
                kernel_set += kernel
        kernel_set += Noise(self.get_input_dims(), self.get_output_dims(), self.params[len(self.params)-1]["noise"])
        return kernel_set

class SM_LMC(model):
    """
    Spectral Mixture - Linear Model of Coregionalization kernel
    with Q components and Rq latent functions.

    Args:
        data (list of mogptk.Data): List of channels, each element should be
            a mogptk.Data object.
        Q (int): Number of components.
        Rq (int): Sub components por components.
        name (str): Name of the model.

    Atributes:
        data (list of mogptk.Data): List of channels, each element is 
            a mogptk.Data object.
        params (dict): Dictionary of lenght Q + 1 with kernel parameters,
            the first Q elements are dicts with parameters of each componentes,
            the last element is the noise variance.

    ----------
    Examples:
    >>> import numpy as np
    >>> t = np.linspace(0, 10, 100)
    >>> y1 = np.sin(0.5 * t)
    >>> y2 = 2 * np.sin(0.2 * t)
    >>> import mogptk
    >>> data_list = []
    >>> mogptk.data_list.append(mogptk.Data(t, y1))
    >>> mogptk.data_list.append(mogptk.Data(t, y2))
    >>> model = mogptk.SM_LMC(data_list, Q=2)
    >>> model.build()
    >>> model.train()
    >>> model.plot_prediction()
    """
    def __init__(self, data, Q=1, Rq=1, name="SM-LMC"):
        model.__init__(self, name, data, Q)

        if Rq != 1:
            raise Exception("Rq != 1 is not (yet) supported") # TODO: support
        self.Rq = Rq
        
        input_dims = self.get_input_dims()
        output_dims = self.get_output_dims()
        for _ in range(Q):
            self.params.append({
                "constant": np.random.standard_normal((Rq, output_dims)),
                "mean": np.random.random((input_dims)),
                "variance": np.random.random((input_dims)),
            })
        self.params.append({
            "noise": np.random.random((output_dims)),
        })
    
    def init_params(self, method='BNSE', sm_init='BNSE', sm_method='BFGS', sm_maxiter=2000, plot=False):
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
            mode (str): Method of initializing, possible values are 'BNSE' and SM.
            sm_init (str): Method of initializing SM kernels. Only valid in 'SM' mode.
            sm_method (str): Optimization method for SM kernels. Only valid in 'SM' mode.
            sm_maxiter (str): Maximum iteration for SM kernels. Only valid in 'SM' mode.
            plt (bool): Show the PSD of the kernel after fitting SM kernels.
                Only valid in 'SM' mode. Default to false.
        """
        # data = self.data.copy()
        # data.normalize()
        
        if method=='BNSE':
            means = np.zeros((self.get_input_dims(), self.Q))
            variances = np.zeros((self.get_input_dims(), self.Q))

            for channel in range(self.get_output_dims()):
                single_amp, single_mean, single_var = self.data[channel].get_bnse_estimation(self.Q)

                means += single_mean
                variances += single_var

                for q in range(self.Q):
                    self.params[q]["constant"][:, channel] = single_amp[:, q].mean()

            # get mean across channels
            means *= 1 / self.get_output_dims()
            variances *= 1 / self.get_output_dims()

            for q in range(self.Q):
                self.params[q]["mean"] = means[:, q] * 2 * np.pi
                self.params[q]["scale"] = variances[:, q] * 2

                # normalize across channels
                self.params[q]["constant"] = np.sqrt(self.params[q]["constant"] / self.params[q]["constant"].mean())
        elif method=='SM':
            # all_params = _estimate_from_sm(self.data, self.Q, init=sm_init, method=sm_method, maxiter=sm_maxiter, plot=plot)

            # input_dims = self.get_input_dims()
            # output_dims = self.get_output_dims()
            # params = {
            #     'weight': np.zeros((self.Q*output_dims)),
            #     'mean': np.zeros((self.Q*output_dims, input_dims)),
            #     'scale': np.zeros((self.Q*output_dims, input_dims)),
            # }
            # for channel in range(output_dims):
            #     for q in range(self.Q):
            #         weight = np.average(all_params[q]['weight'][:,channel])
            #         if weight != 0.0:
            #             weight /= np.sum(all_params[q]['weight'])
            #         mean = all_params[q]['mean'][:,channel].reshape(1, -1)
            #         scale = all_params[q]['scale'][:,channel].reshape(1, -1)
            #         params['weight'][channel*q+q] = weight
            #         params['mean'][channel*q+q,:] = mean
            #         params['scale'][channel*q+q,:] = scale

            # indices = np.argsort(params['weight'])[::-1]
            # for q in range(self.Q):
            #     if q < len(indices):
            #         i = indices[q]
            #         self.params[q]['mean'] = params['mean'][i]
            #         self.params[q]['variance'] = params['scale'][i] * 2
            params = _estimate_from_sm(self.data, self.Q, init=sm_init, method=sm_method, maxiter=sm_maxiter, plot=plot)
            for q in range(self.Q):
                self.params[q]["constant"] = params[q]['weight'].mean(axis=0).reshape(self.Rq, -1)
                self.params[q]["mean"] = params[q]['mean'].mean(axis=1)
                self.params[q]["variance"] = params[q]['scale'].mean(axis=1) * 2
        else:
            raise Exception("possible modes are either 'BNSE' or 'SM'")

        # noise init
        self.params[self.Q]['noise'] = _estimate_noise_var(self.data)

    def _transform_data(self, x, y=None):
        return _transform_data_mogp(x, y)

    def _kernel(self):
        for q in range(self.Q):
            kernel = SpectralMixtureLMC(
                self.get_input_dims(),
                self.get_output_dims(),
                self.Rq,
                self.params[q]["constant"],
                self.params[q]["mean"],
                self.params[q]["variance"],
            )
            if q == 0:
                kernel_set = kernel
            else:
                kernel_set += kernel
        kernel_set += Noise(self.get_input_dims(), self.get_output_dims(), self.params[len(self.params)-1]["noise"])
        return kernel_set

class CG(model):
    """
    CG is the Convolutional Gaussian kernel with Q components [1].

    Args:
        data (list of mogptk.Data): List of channels, each element should be
            a mogptk.Data object.
        Q (int): Number of components.
        name (str): Name of the model.

    Atributes:
        data (list of mogptk.Data): List of channels, each element is 
            a mogptk.Data object.
        Q (int): Number of components.
        params (dict): Dictionary of lenght Q + 1 with kernel parameters,
            the first Q elements are dicts with parameters of each components,
            the last element is the noise variance.

    ----------
    Examples:
    >>> import numpy as np
    >>> t = np.linspace(0, 10, 100)
    >>> y1 = np.sin(0.5 * t)
    >>> y2 = 2 * np.sin(0.2 * t)
    >>> import mogptk
    >>> data_list = []
    >>> mogptk.data_list.append(mogptk.Data(t, y1))
    >>> mogptk.data_list.append(mogptk.Data(t, y2))
    >>> model = mogptk.CG(data_list, Q=2)
    >>> model.build()
    >>> model.train()
    >>> model.plot_prediction()

    References:
    [1] M. M. Alvarez and N. Lawrence, "Sparse convolved multiple output gaussian processes", 2009.
    """
    def __init__(self, data, Q=1, name="CG"):
        model.__init__(self, name, data, Q)
        
        input_dims = self.get_input_dims()
        output_dims = self.get_output_dims()
        for _ in range(Q):
            self.params.append({
                "constant": np.random.random((output_dims)),
                "variance": np.random.random((input_dims, output_dims)),
                "latent_variance": np.random.random((input_dims)),
            })
        self.params.append({
            "noise": np.random.random((output_dims)),
        })

    def init_params(self, sm_init='random', sm_method='BFGS', sm_maxiter=2000, plot=False):
        """
        Initialize kernel parameters, variance and mixture weights. 

        The initialization is done fitting a single output GP with Sepectral mixture (SM)
        kernel for each channel with spectral means fixed to 0 for all Q.

        In all cases the noise is initialized with 1/30 of the variance 
        for each channel.

        Args:
            sm_init (str): Method of initializing SM kernels.
            sm_method (str): Optimization method for SM kernels.
            sm_maxiter (str): Maximum iteration for SM kernels.
            plt (bool): Show the PSD of the kernel after fitting SM kernels.
                Default to false.
        """
        params = _estimate_from_sm(self.data,
            self.Q,
            init=sm_init,
            method=sm_method,
            maxiter=sm_maxiter,
            plot=plot,
            fix_means=True)

        for q in range(self.Q):
            self.params[q]["variance"] = params[q]['scale']
            self.params[q]["constant"] = params[q]['weight'].mean(axis=0)

            self.params[q]["constant"] = self.params[q]["constant"] / self.params[q]["constant"].mean()

        # noise init
        self.params[self.Q]['noise'] = _estimate_noise_var(self.data)

    def _transform_data(self, x, y=None):
        return _transform_data_mogp(x, y)

    def _kernel(self):
        for q in range(self.Q):
            kernel = ConvolutionalGaussian(
                self.get_input_dims(),
                self.get_output_dims(),
                self.params[q]["constant"],
                self.params[q]["variance"],
                self.params[q]["latent_variance"],
            )
            if q == 0:
                kernel_set = kernel
            else:
                kernel_set += kernel
        kernel_set += Noise(self.get_input_dims(), self.get_output_dims(), self.params[len(self.params)-1]["noise"])
        return kernel_set
    
