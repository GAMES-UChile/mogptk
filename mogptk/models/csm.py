import numpy as np

from ..dataset import DataSet
from ..model import Model, Exact, logger
from ..gpr import CrossSpectralKernel, MixtureKernel, GaussianLikelihood

class CSM(Model):
    """
    Cross Spectral Mixture kernel [1] with `Q` components and `Rq` latent functions. The parameters will be randomly instantiated, use `init_parameters()` to initialize the parameters to reasonable values for the current data set.

    Args:
        dataset (mogptk.dataset.DataSet): `DataSet` object of data for all channels.
        Q (int): Number of components.
        Rq (int): Number of subcomponents.
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
    >>> y1 = np.sin(0.5 * t)
    >>> y2 = 2.0 * np.sin(0.2 * t)
    >>> 
    >>> dataset = mogptk.DataSet(t, [y1, y2])
    >>> model = mogptk.CSM(dataset, Q=2)
    >>> model.init_parameters()
    >>> model.train()
    >>> model.predict()
    >>> dataset.plot()

    [1] K.R. Ulrich et al, "GP Kernels for Cross-Spectrum Analysis", Advances in Neural Information Processing Systems 28, 2015
    """
    def __init__(self, dataset, Q=1, Rq=1, inference=Exact(), mean=None, name="CSM"):
        if not isinstance(dataset, DataSet):
            dataset = DataSet(dataset)

        output_dims = dataset.get_output_dims()
        input_dims = dataset.get_input_dims()[0]
        for input_dim in dataset.get_input_dims()[1:]:
            if input_dim != input_dims:
                raise ValueError("input dimensions for all channels must match")

        spectral = CrossSpectralKernel(output_dims=output_dims, input_dims=input_dims, Rq=Rq)
        kernel = MixtureKernel(spectral, Q)
        for q in range(Q):
            kernel[q].amplitude.assign(np.random.rand(output_dims,Rq))
            kernel[q].mean.assign(np.random.rand(input_dims))
            kernel[q].variance.assign(np.random.rand(input_dims))

        super().__init__(dataset, kernel, inference, mean, name)
        self.Q = Q
        self.Rq = Rq
        nyquist = np.amin(self.dataset.get_nyquist_estimation(), axis=0)
        for q in range(Q):
            self.gpr.kernel[q].mean.assign(upper=nyquist)
    
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

        output_dims = self.dataset.get_output_dims()
        means = np.concatenate(means, axis=0)
        variances = np.concatenate(variances, axis=0)
        constant = np.random.random((output_dims, self.Q, self.Rq))
        for q in range(self.Q):
            for j in range(len(self.dataset)):
                constant[j,q,:] = amplitudes[j][q,:].mean()**2 / self.Rq
            self.gpr.kernel[q].amplitude.assign(constant[:,q,:])
            self.gpr.kernel[q].mean.assign(means[q,:])
            self.gpr.kernel[q].variance.assign(variances[q,:])

        # noise
        if isinstance(self.gpr.likelihood, GaussianLikelihood):
            _, Y = self.dataset.get_train_data(transformed=True)
            Y_std = [Y[j].std() for j in range(self.dataset.get_output_dims())]
            if self.gpr.likelihood.scale().ndim == 0:
                self.gpr.likelihood.scale.assign(np.mean(Y_std))
            else:
                self.gpr.likelihood.scale.assign(Y_std)
