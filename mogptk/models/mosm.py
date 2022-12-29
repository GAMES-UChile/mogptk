import numpy as np
import matplotlib.pyplot as plt

from ..dataset import DataSet
from ..model import Model, Exact, logger
from ..gpr import MultiOutputSpectralMixtureKernel, GaussianLikelihood
from ..util import plot_spectrum

class MOSM(Model):
    """
    Multi-Output Spectral Mixture kernel with `Q` components as proposed by [1]. The parameters will be randomly instantiated, use `init_parameters()` to initialize the parameters to reasonable values for the current data set.

    Args:
        dataset (mogptk.dataset.DataSet): `DataSet` object of data for all channels.
        Q (int): Number of components.
        inference: Gaussian process inference model to use, such as `mogptk.Exact`.
        mean (mogptk.gpr.mean.Mean): The mean class.
        name (str): Name of the model.

    Atributes:
        dataset: The associated mogptk.dataset.DataSet.
        gpr: The mogptk.gpr.model.Model.

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
    def __init__(self, dataset, Q=1, inference=Exact(), mean=None, name="MOSM"):
        if not isinstance(dataset, DataSet):
            dataset = DataSet(dataset)

        output_dims = dataset.get_output_dims()
        input_dims = dataset.get_input_dims()[0]
        for input_dim in dataset.get_input_dims()[1:]:
            if input_dim != input_dims:
                raise ValueError("input dimensions for all channels must match")

        kernel = MultiOutputSpectralMixtureKernel(Q=Q, output_dims=output_dims, input_dims=input_dims)
        kernel.weight.assign(np.random.rand(output_dims,Q))
        kernel.mean.assign(np.random.rand(output_dims,Q,input_dims))
        kernel.variance.assign(np.random.rand(output_dims,Q,input_dims))

        super().__init__(dataset, kernel, inference, mean, name)
        self.Q = Q
        nyquist = np.array(self.dataset.get_nyquist_estimation())[:,None,:].repeat(Q,axis=1)
        self.gpr.kernel.mean.assign(upper=nyquist)

    def init_parameters(self, method='BNSE', iters=500):
        """
        Estimate kernel parameters from the data set. The initialization can be done using three methods:

        - BNSE estimates the PSD via Bayesian non-parametris spectral estimation (Tobar 2018) and then selecting the greater Q peaks in the estimated spectrum, and use the peak's position, magnitude and width to initialize the mean, magnitude and variance of the kernel respectively.
        - LS is similar to BNSE but uses Lomb-Scargle to estimate the spectrum, which is much faster but may give poorer results.
        - SM fits independent Gaussian processes for each channel, each one with a spectral mixture kernel, and uses the fitted parameters as initial values for the multi-output kernel.

        In all cases the noise is initialized with 1/30 of the variance of each channel.

        Args:
            method (str): Method of estimation, such as BNSE, LS, or SM.
            iters (str): Number of iterations for initialization.
        """

        input_dims = self.dataset.get_input_dims()
        output_dims = self.dataset.get_output_dims()

        if not method.lower() in ['bnse', 'ls', 'sm']:
            raise ValueError("valid methods of estimation are BNSE, LS, and SM")

        if method.lower() == 'bnse':
            amplitudes, means, variances = self.dataset.get_bnse_estimation(self.Q, iters=iters)
        elif method.lower() == 'ls':
            amplitudes, means, variances = self.dataset.get_ls_estimation(self.Q)
        else:
            amplitudes, means, variances = self.dataset.get_sm_estimation(self.Q, iters=iters)
        if len(amplitudes) == 0:
            logger.warning('{} could not find peaks for MOSM'.format(method))
            return

        weight = np.zeros((output_dims,self.Q))
        mean = np.zeros((output_dims,self.Q,input_dims[0]))
        variance = np.zeros((output_dims,self.Q,input_dims[0]))
        for q in range(self.Q):
            for j in range(output_dims):
                weight[j,q] = 10.0*amplitudes[j][q,:].mean()
                mean[j,q,:] = means[j][q,:]
                variance[j,q,:] = variances[j][q,:]

        self.gpr.kernel.weight.assign(weight)
        self.gpr.kernel.mean.assign(mean)
        self.gpr.kernel.variance.assign(variance)

        # noise
        if isinstance(self.gpr.likelihood, GaussianLikelihood):
            _, Y = self.dataset.get_train_data(transformed=True)
            Y_std = [Y[j].std() for j in range(self.dataset.get_output_dims())]
            if self.gpr.likelihood.scale().ndim == 0:
                self.gpr.likelihood.scale.assign(np.mean(Y_std))
            else:
                self.gpr.likelihood.scale.assign(Y_std)

    def check(self):
        """
        Check validity of model and parameters.
        """
        for j in range(self.dataset.get_output_dims()):
            for q in range(self.Q):
                mean = self.gpr.kernel.mean.numpy()[j,q,:]
                var = self.gpr.kernel.variance.numpy()[j,q,:]
                if np.linalg.norm(mean) < np.linalg.norm(var):
                    print("‣ MOSM approaches RBF kernel for q=%d in channel='%s'" % (q, self.dataset[j].name))

    def plot_spectrum(self, method='LS', maxfreq=None, log=False, noise=False, title=None):
        """
        Plot spectrum of kernel.

        Args:
            method (str): Set the method to get the spectrum from the data such as LS or BNSE.
            maxfreq (float): Maximum frequency to plot, otherwise the Nyquist frequency is used.
            log (boolean): Show X and Y axis in log-scale.
            noise (boolean): Add noise to the PSD.
            title (str): Set the title of the plot.

        Returns:
            matplotlib.figure.Figure: Figure.
            matplotlib.axes.Axes: Axes.
        """
        input_dims = self.dataset.get_input_dims()[0]
        names = self.dataset.get_names()
        if maxfreq is not None:
            maxfreq = [maxfreq] * len(self.dataset)
        means = self.gpr.kernel.mean.numpy().transpose([1,0,2])
        scales = np.sqrt(self.gpr.kernel.variance.numpy().transpose([1,0,2]))
        weights = self.gpr.kernel.weight.numpy().transpose([1,0])**2

        noises = None
        if noise:
            if not isinstance(self.gpr.likelihood, GaussianLikelihood):
                raise ValueError("likelihood must be Gaussian to enable spectral noise")
            if isinstance(self.gpr, Exact) and self.variance_per_data:
                raise ValueError("likelihood variance must not be per data point to enable spectral noise")
            noises = self.gpr.likelihood.scale.numpy()

        return plot_spectrum(means, scales, dataset=self.dataset, weights=weights, noises=noises, method=method, maxfreq=maxfreq, log=log, titles=names, title=title)

    def plot_cross_spectrum(self, title=None, figsize=(12,12)):
        """
        Plot power spectral density and power cross spectral density.

        Args:
            title (str): Set the title of the plot.

        Returns:
            matplotlib.figure.Figure: Figure.
            matplotlib.axes.Axes: Axes.
        """

        if not all(input_dims == 1 for input_dims in self.dataset.get_input_dims()):
            raise RuntimeError("not implemented for multiple input dimensions")

        input_dims = self.dataset.get_input_dims()[0]
        output_dims = self.dataset.get_output_dims()
        Q = self.Q

        cross_params = {}
        cross_params['covariance'] = np.zeros((output_dims, output_dims, input_dims, Q))
        cross_params['mean'] = np.zeros((output_dims, output_dims, input_dims, Q))
        cross_params['magnitude'] = np.zeros((output_dims, output_dims, Q))
        cross_params['delay'] = np.zeros((output_dims, output_dims, input_dims, Q))
        cross_params['phase'] = np.zeros((output_dims, output_dims, Q))

        weight = self.gpr.kernel.weight.numpy()
        mean = self.gpr.kernel.mean.numpy()
        variance = self.gpr.kernel.variance.numpy()
        phase = self.gpr.kernel.phase.numpy()
        delay = self.gpr.kernel.delay.numpy()
        for q in range(Q):
            for i in range(output_dims):
                for j in range(output_dims):
                    w_i = weight[i,q]
                    w_j = weight[j,q]
                    mu_i = mean[i,q,:]
                    mu_j = mean[j,q,:]
                    var_i = variance[i,q,:]
                    var_j = variance[j,q,:]
                    sv = var_i + var_j

                    cross_params['covariance'][i, j, :, q] = 2 * (var_i * var_j) / sv
                    cross_mean_num = var_i.dot(mu_j) + var_j.dot(mu_i)
                    cross_params['mean'][i, j, :, q] = cross_mean_num / sv
                    exp_term = -1/4 * ((mu_i - mu_j)**2 / sv).sum()
                    cross_params['magnitude'][i, j, q] = w_i * w_j * np.exp(exp_term)
                    for k in range(input_dims):
                        cross_params['delay'][i, j, k, q] = delay[i,q,k]-delay[j,q,k]
                    cross_params['phase'][i, j, q] = phase[i,q]-phase[j,q]

        h = figsize[1]
        fig, axes = plt.subplots(output_dims, output_dims, figsize=figsize, squeeze=False, constrained_layout=True)
        if title is not None:
            fig.suptitle(title, y=(h+0.8)/h, fontsize=18)

        for j in range(output_dims):
            for i in range(j+1):
                magn = cross_params['magnitude'][j, i, :]
                mean = cross_params['mean'][j, i, 0, :]
                cov = cross_params['covariance'][j, i, 0, :]
                delay = cross_params['delay'][j, i, 0, :]
                phase = cross_params['phase'][j, i, :]

                w_high = (mean + 2* np.sqrt(cov)).max()
                w = np.linspace(-w_high, w_high, 1000)
                if i == j:
                    # power spectral density
                    psd_total = np.zeros(len(w))
                    for q in range(self.Q):
                        psd_q = np.exp(-0.5 * (w - mean[q])**2 / cov[q])
                        psd_q += np.exp(-0.5 * (w + mean[q])**2 / cov[q])
                        psd_q *= magn[q] * 0.5
                        axes[j,i].plot(w, psd_q, ls='--', c='k')
                        psd_total += psd_q
                    axes[j,i].plot(w, psd_total, c='k')
                else:
                    # power cross spectral density
                    psd_total = np.zeros(len(w)) + 0.j
                    for q in range(self.Q):
                        psd_q = np.exp(-0.5 * (w - mean[q])**2 / cov[q] + 1.j * (w * delay[q] + phase[q]))
                        psd_q += np.exp(-0.5 * (w + mean[q])**2 / cov[q] + 1.j * (w * delay[q] + phase[q]))
                        psd_q *= magn[q] * 0.5
                        axes[j,i].plot(w, np.real(psd_q), ls='--', c='k')
                        axes[j,i].plot(w, np.imag(psd_q), ls='--', c='silver')
                        psd_total += psd_q
                    axes[j,i].plot(w, np.real(psd_total), c='k')
                    axes[j,i].plot(w, np.imag(psd_total), c='silver')
                axes[j,i].set_yticks([])
            for i in range(j+1, output_dims):
                axes[j,i].set_axis_off()

        legends = []
        legends.append(plt.Line2D([0], [0], ls='-', color='k', label='Total (real)'))
        legends.append(plt.Line2D([0], [0], ls='--', color='k', label='Mixture (real)'))
        legends.append(plt.Line2D([0], [0], ls='-', color='silver', label='Total (imag)'))
        legends.append(plt.Line2D([0], [0], ls='--', color='silver', label='Mixture (imag)'))
        fig.legend(handles=legends)
        return fig, axes
