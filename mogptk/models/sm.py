import numpy as np

from ..dataset import DataSet
from ..model import Model, Exact, logger
from ..gpr import SpectralMixtureKernel, IndependentMultiOutputKernel, GaussianLikelihood
from ..util import plot_spectrum

class SM(Model):
    """
    Independent Spectral Mixture kernels per channel. The spectral mixture kernel is proposed by [1]. The parameters will be randomly instantiated, use `init_parameters()` to initialize the parameters to reasonable values for the current data set.

    Args:
        dataset (mogptk.dataset.DataSet): `DataSet` object of data for all channels.
        Q (int): Number of components.
        inference: Gaussian process inference model to use, such as `mogptk.Exact`.
        mean (mogptk.gpr.mean.Mean): The mean class.
        name (str): Name of the model.

    Attributes:
        dataset: The associated mogptk.dataset.DataSet.
        gpr: The mogptk.gpr.model.Model.

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
    def __init__(self, dataset, Q=1, inference=Exact(), mean=None, name="SM"):
        if not isinstance(dataset, DataSet):
            dataset = DataSet(dataset)

        output_dims = dataset.get_output_dims()
        input_dims = dataset.get_input_dims()[0]
        kernel = IndependentMultiOutputKernel(
            [SpectralMixtureKernel(Q=Q, input_dims=input_dims)
                for j in range(output_dims)],
            output_dims=output_dims)
        for j in range(output_dims):
            kernel[j].magnitude.assign(np.random.rand(Q))
            kernel[j].mean.assign(np.random.rand(Q,input_dims))
            kernel[j].variance.assign(np.random.rand(Q,input_dims))

        super().__init__(dataset, kernel, inference, mean, name)
        self.Q = Q
        nyquist = np.array(self.dataset.get_nyquist_estimation())[:,None,:].repeat(Q,axis=1)
        for j in range(output_dims):
            self.gpr.kernel[j].mean.assign(upper=nyquist[j,:,:])

    def init_parameters(self, method='LS', iters=500):
        """
        Estimate kernel parameters from the data set. The initialization can be done using three methods:

        - BNSE estimates the PSD via Bayesian non-parametris spectral estimation (Tobar 2018) and then selecting the greater Q peaks in the estimated spectrum, and use the peak's position, magnitude and width to initialize the mean, magnitude and variance of the kernel respectively.
        - LS is similar to BNSE but uses Lomb-Scargle to estimate the spectrum, which is much faster but may give poorer results.
        - IPS uses independent parameter sampling from the PhD thesis of Andrew Wilson 2014. It takes the inverse of the lengthscales drawn from a truncated Gaussian Normal(0, max_dist^2), the means drawn from a Unif(0, 0.5 / minimum distance between two points), and the mixture weights by taking the standard variation of the Y values divided by the number of mixtures.

        In all cases the noise is initialized with 1/30 of the variance of each channel.

        Args:
            method (str): Method of estimation, such as IPS, LS, or BNSE.
            iters (str): Number of iterations for initialization.
        """

        input_dims = self.dataset.get_input_dims()
        output_dims = self.dataset.get_output_dims()

        if method.lower() not in ['ips', 'ls', 'bnse']:
            raise ValueError("valid methods of estimation are IPS, LS, and BNSE")

        if method.lower() == 'ips':
            for j in range(output_dims):
                nyquist = self.dataset[j].get_nyquist_estimation()
                x = self.dataset[j].X[self.dataset[j].mask,:]
                y = self.dataset[j].Y_transformer.forward(self.dataset[j].Y[self.dataset[j].mask], x)
                x_range = np.max(x, axis=0) - np.min(x, axis=0)

                weights = [2.0*y.std()/self.Q] * self.Q
                means = nyquist * np.random.rand(self.Q, input_dims[j])
                variances = 1.0 / (np.abs(np.random.randn(self.Q, input_dims[j])) * x_range)

                self.gpr.kernel[j].magnitude.assign(weights)
                self.gpr.kernel[j].mean.assign(means)
                self.gpr.kernel[j].variance.assign(variances)
            return
        elif method.lower() == 'ls':
            amplitudes, means, variances = self.dataset.get_ls_estimation(self.Q)
            if len(amplitudes) == 0:
                logger.warning('LS could not find peaks for SM')
                return
        elif method.lower() == 'bnse':
            amplitudes, means, variances = self.dataset.get_bnse_estimation(self.Q, iters=iters)
            if np.sum(amplitudes) == 0.0:
                logger.warning('BNSE could not find peaks for SM')
                return

        for j in range(output_dims):
            self.gpr.kernel[j].magnitude.assign(amplitudes[j].mean(axis=1)**2)
            self.gpr.kernel[j].mean.assign(means[j])
            self.gpr.kernel[j].variance.assign(variances[j])

        # noise
        if isinstance(self.gpr.likelihood, GaussianLikelihood):
            _, Y = self.dataset.get_train_data(transformed=True)
            Y_std = [Y[j].std() for j in range(self.dataset.get_output_dims())]
            if self.gpr.likelihood.scale().ndim == 0:
                self.gpr.likelihood.scale.assign(np.mean(Y_std))
            else:
                self.gpr.likelihood.scale.assign(Y_std)

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
        output_dims = self.dataset.get_output_dims()
        names = self.dataset.get_names()
        if maxfreq is not None:
            maxfreq = [maxfreq] * len(self.dataset)
        means = np.array([self.gpr.kernel[j].mean.numpy() for j in range(output_dims)]).transpose([1,0,2])
        scales = np.array([np.sqrt(self.gpr.kernel[j].variance.numpy()) for j in range(output_dims)]).transpose([1,0,2])
        weights = np.array([self.gpr.kernel[j].magnitude.numpy() for j in range(output_dims)]).transpose([1,0])

        noises = None
        if noise:
            if not isinstance(self.gpr.likelihood, GaussianLikelihood):
                raise ValueError("likelihood must be Gaussian to enable spectral noise")
            if isinstance(self.gpr, Exact) and self.variance_per_data:
                raise ValueError("likelihood variance must not be per data point to enable spectral noise")
            noises = self.gpr.likelihood.scale.numpy()

        return plot_spectrum(means, scales, dataset=self.dataset, weights=weights, noises=noises, method=method, maxfreq=maxfreq, log=log, titles=names, title=title)
