import numpy as np
import matplotlib.pyplot as plt

from .dataset import DataSet
from .model import Model, Exact, logger
from .kernels import MultiOutputSpectralKernel, MixtureKernel
from .plot import plot_spectrum

class MOSM(Model):
    """
    Multi-Output Spectral Mixture kernel with `Q` components as proposed by [1]. The parameters will be randomly instantiated, use `init_parameters()` to initialize the parameters to reasonable values for the current data set.

    Args:
        dataset (mogptk.dataset.DataSet): `DataSet` object of data for all channels.
        Q (int): Number of components.
        model: Gaussian process model to use, such as `mogptk.model.Exact`.
        mean (mogptk.kernels.mean.Mean): The mean class.
        name (str): Name of the model.

    Atributes:
        dataset: The associated mogptk.dataset.DataSet.
        model: The mogptk.kernels.model.Model.
        kernel: The mogptk.kernels.kernel.Kernel.

    Examples:
    >>> import numpy as np
    >>> import mogptk
    >>> 
    >>> t = np.linspace(0, 10, 100)
    >>> y1 = np.sin(0.5 * t)
    >>> y2 = 2.0 * np.sin(0.2 * t)
    >>> 
    >>> dataset = mogptk.DataSet(t, [y1, y2])
    >>> model = mogptk.MOSM(dataset, Q=2)
    >>> model.init_parameters()
    >>> model.train()
    >>> model.predict()
    >>> dataset.plot()

    [1] G. Parra and F. Tobar, "Spectral Mixture Kernels for Multi-Output Gaussian Processes", Advances in Neural Information Processing Systems 31, 2017
    """
    def __init__(self, dataset, Q=1, model=Exact(), mean=None, name="MOSM"):
        if not isinstance(dataset, DataSet):
            dataset = DataSet(dataset)
        dataset.rescale_x()

        spectral = MultiOutputSpectralKernel(
            output_dims=dataset.get_output_dims(),
            input_dims=dataset.get_input_dims()[0],
        )
        kernel = MixtureKernel(spectral, Q)

        super(MOSM, self).__init__(dataset, kernel, model, mean, name)
        self.Q = Q

    def init_parameters(self, method='BNSE', sm_init='BNSE', sm_method='Adam', sm_iters=100, sm_params={}, sm_plot=False):
        """
        Estimate kernel parameters from the data set. The initialization can be done using three methods:

        - BNSE estimates the PSD via Bayesian non-parametris spectral estimation (Tobar 2018) and then selecting the greater Q peaks in the estimated spectrum, and use the peak's position, magnitude and width to initialize the mean, magnitude and variance of the kernel respectively.
        - LS is similar to BNSE but uses Lomb-Scargle to estimate the spectrum, which is much faster but may give poorer results.
        - SM fits independent Gaussian processes for each channel, each one with a spectral mixture kernel, and uses the fitted parameters as initial values for the multi-output kernel.

        In all cases the noise is initialized with 1/30 of the variance of each channel.

        Args:
            method (str): Method of estimation, such as BNSE, LS, or SM.
            sm_init (str): Parameter initialization strategy for SM initialization.
            sm_method (str): Optimization method for SM initialization.
            sm_iters (str): Number of iterations for SM initialization.
            sm_params (object): Additional parameters for PyTorch optimizer.
            sm_plot (bool): Show the PSD of the kernel after fitting SM.
        """

        input_dims = self.dataset.get_input_dims()
        output_dims = self.dataset.get_output_dims()

        if not method.lower() in ['bnse', 'ls', 'sm']:
            raise ValueError("valid methods of estimation are BNSE, LS, and SM")

        if method.lower() == 'bnse':
            amplitudes, means, variances = self.dataset.get_bnse_estimation(self.Q)
        elif method.lower() == 'ls':
            amplitudes, means, variances = self.dataset.get_lombscargle_estimation(self.Q)
        else:
            amplitudes, means, variances = self.dataset.get_sm_estimation(self.Q, method=sm_init, optimizer=sm_method, iters=sm_iters, params=sm_params, plot=sm_plot)
        if len(amplitudes) == 0:
            logger.warning('{} could not find peaks for MOSM'.format(method))
            return

        magnitude = np.zeros((output_dims, self.Q))
        for q in range(self.Q):
            mean = np.zeros((output_dims,input_dims[0]))
            variance = np.zeros((output_dims,input_dims[0]))
            for j in range(output_dims):
                if q < amplitudes[j].shape[0]:
                    magnitude[j,q] = amplitudes[j][q,:].mean()
                    mean[j,:] = means[j][q,:]
                    # maybe will have problems with higher input dimensions
                    variance[j,:] = variances[j][q,:] * (4 + 20 * (max(input_dims) - 1)) # 20
            self.model.kernel[q].mean.assign(mean)
            self.model.kernel[q].variance.assign(variance)

        # normalize proportional to channels variances
        for j, channel in enumerate(self.dataset):
            _, y = channel.get_train_data(transformed=True)
            if 0.0 < magnitude[j,:].sum():
                magnitude[j,:] = np.sqrt(magnitude[j,:] / magnitude[j,:].sum() * y.var()) * 2
        
        for q in range(self.Q):
            self.model.kernel[q].magnitude.assign(magnitude[:,q])

        noise = np.empty((output_dims,))
        for j, channel in enumerate(self.dataset):
            _, y = channel.get_train_data(transformed=True)
            noise[j] = y.var() / 30.0
        for q in range(self.Q):
            self.model.kernel[q].noise.assign(noise)

    def check(self):
        """
        Check validity of model and parameters.
        """
        for j in range(self.dataset.get_output_dims()):
            for q in range(self.Q):
                mean = self.model.kernel[q].mean.numpy()[j,:]
                var = self.model.kernel[q].variance.numpy()[j,:]
                if np.linalg.norm(mean) < np.linalg.norm(var):
                    print("‣ MOSM approaches RBF kernel for q=%d in channel='%s'" % (q, self.dataset[j].name))

    def plot_spectrum(self, title=None):
        """
        Plot spectrum of kernel.
        """
        names = self.dataset.get_names()
        nyquist = self.dataset.get_nyquist_estimation()

        means = np.array([self.model.kernel[q].mean.numpy() for q in range(self.Q)])
        scales = np.array([self.model.kernel[q].variance.numpy() for q in range(self.Q)])
        weights = np.array([self.model.kernel[q].magnitude.numpy() for q in range(self.Q)])**2

        return plot_spectrum(means, scales, weights=weights, nyquist=nyquist, titles=names, title=title)

    def plot_cross_spectrum(self, title=None, figsize=(12,12)):
        """
        Plot power spectral density and power cross spectral density.
        """

        if not all(input_dims == 1 for input_dims in self.dataset.get_input_dims()):
            raise RuntimeError("not implemented for multiple input dimensions")

        cross_params = self._get_cross_parameters()
        output_dims = self.dataset.get_output_dims()

        h = figsize[1]
        fig, axes = plt.subplots(output_dims, output_dims, figsize=figsize, squeeze=False, constrained_layout=True)
        if title is not None:
            fig.suptitle(title, y=(h+0.8)/h, fontsize=18)

        for j in range(output_dims):
            for i in range(j+1):
                self._plot_power_cross_spectral_density(axes[j,i], cross_params, channels=(j,i))
            for i in range(j+1, output_dims):
                axes[j,i].set_axis_off()

        legends = []
        legends.append(plt.Line2D([0], [0], ls='-', color='k', label='Total (real)'))
        legends.append(plt.Line2D([0], [0], ls='--', color='k', label='Mixture (real)'))
        legends.append(plt.Line2D([0], [0], ls='-', color='silver', label='Total (imag)'))
        legends.append(plt.Line2D([0], [0], ls='--', color='silver', label='Mixture (imag)'))
        fig.legend(handles=legends, loc="upper center", bbox_to_anchor=(0.5,(h+0.4)/h), ncol=5)

        return fig, axes

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

        w_high = (mean + 2* np.sqrt(cov)).max()
        w = np.linspace(-w_high, w_high, 1000)

        # power spectral density
        if i == j:
            psd_total = np.zeros(len(w))
            for q in range(self.Q):
                psd_q = np.exp(-0.5 * (w - mean[q])**2 / cov[q])
                psd_q += np.exp(-0.5 * (w + mean[q])**2 / cov[q])
                psd_q *= magn[q] * 0.5
                ax.plot(w, psd_q, ls='--', c='k')
                psd_total += psd_q
            ax.plot(w, psd_total, c='k')
        # power cross spectral density
        else:
            psd_total = np.zeros(len(w)) + 0.j
            for q in range(self.Q):
                psd_q = np.exp(-0.5 * (w - mean[q])**2 / cov[q] + 1.j * (w * delay[q] + phase[q]))
                psd_q += np.exp(-0.5 * (w + mean[q])**2 / cov[q] + 1.j * (w * delay[q] + phase[q]))
                psd_q *= magn[q] * 0.5
                ax.plot(w, np.real(psd_q), ls='--', c='k')
                ax.plot(w, np.imag(psd_q), ls='--', c='silver')
                psd_total += psd_q
            ax.plot(w, np.real(psd_total), c='k')
            ax.plot(w, np.imag(psd_total), c='silver')
        ax.set_yticks([])
        return

    def _get_cross_parameters(self):
        """
        Obtain cross parameters from MOSM.

        Returns:
            cross_params(dict): Dictionary with the cross parameters: covariance, mean, magnitude, delay and phase. Each one an array of shape (output_dim,output_dim,input_dim,Q) with the cross parameters, with the exception of magnitude and phase where the cross parameters are of shape (output_dim,output_dim,Q).

        This assumes the same input dimension for all channels.
        """
        m = self.dataset.get_output_dims()
        d = self.dataset.get_input_dims()[0]
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
                    magnitude = self.model.kernel[q].magnitude.numpy()
                    mean = self.model.kernel[q].mean.numpy()
                    variance = self.model.kernel[q].variance.numpy()

                    w_i = magnitude[i]
                    w_j = magnitude[j]
                    mu_i = mean[i,:]
                    mu_j = mean[j,:]
                    var_i = variance[i,:]
                    var_j = variance[j,:]
                    sv = var_i + var_j

                    # cross covariance
                    cross_params['covariance'][i, j, :, q] = 2 * (var_i * var_j) / sv
                    # cross mean
                    cross_mean_num = var_i.dot(mu_j) + var_j.dot(mu_i)
                    cross_params['mean'][i, j, :, q] = cross_mean_num / sv
                    # cross magnitude
                    exp_term = -1/4 * ((mu_i - mu_j)**2 / sv).sum()
                    cross_params['magnitude'][i, j, q] = w_i * w_j * np.exp(exp_term)
            if m>1:
                # cross phase
                phase_q = self.model.kernel[q].phase.numpy()
                cross_params['phase'][:, :, q] = np.subtract.outer(phase_q, phase_q)
                for n in range(d):
                    # cross delay        
                    delay_n_q = self.model.kernel[q].delay.numpy()[:,n]
                    cross_params['delay'][:, :, n, q] = np.subtract.outer(delay_n_q, delay_n_q)

        return cross_params
