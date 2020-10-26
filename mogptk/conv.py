import numpy as np

from .model import model, logger
from .kernels import GaussianConvolutionProcessKernel, MixtureKernel

class CONV(model):
    """
    CONV is the Convolutional Gaussian kernel with Q components [1].

    Args:
        dataset (mogptk.dataset.DataSet): DataSet object of data for all channels.
        Q (int, optional): Number of components.
        name (str, optional): Name of the model.
        likelihood (gpflow.likelihoods, optional): Likelihood to use from GPFlow, if None a default exact inference Gaussian likelihood is used.
        variational (bool, optional): If True, use variational inference to approximate function values as Gaussian. If False it will use Monte Carlo Markov Chain.
        sparse (bool), optional: If True, will use sparse GP regression.
        like_params (dict, optional): Parameters to GPflow likelihood.

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

    [1] M.A. Álvarez and N.D. Lawrence, "Sparse Convolved Multiple Output Gaussian Processes", Advances in Neural Information Processing Systems 21, 2009
    """
    def __init__(self, dataset, Q=1, name="CONV"):
        if len(dataset)<2:
            raise Exception("Number of channels equal 1, model CONV must be used with at least 2, for 1 channel use SM instead.")
        model.__init__(self, name, dataset)
        self.Q = Q

        input_dims = self.dataset.get_input_dims()
        for input_dim in input_dims[1:]:
            if input_dim != input_dims[0]:
                raise ValueError("input dimensions for all channels must match")

        conv = GaussianConvolutionProcessKernel(
            output_dims=self.dataset.get_output_dims(),
            input_dims=self.dataset.get_input_dims()[0],
        )
        kernel = MixtureKernel(conv, Q=Q)
        self._build(kernel)
        self.model.noise.assign(0.0, lower=0.0, trainable=False)  # handled by MultiOutputKernel

    def init_parameters(self, method='SM', sm_method='BNSE', sm_opt='LBFGS', sm_maxiter=2000, plot=False):
        """
        Initialize kernel parameters, variance and mixture weights. 

        The initialization is done fitting a single output GP with Sepectral mixture (SM)
        kernel for each channel with spectral means fixed to 0 for all Q.

        In all cases the noise is initialized with 1/30 of the variance 
        for each channel.

        Args:
            method (str, optional): Method of estimation, such as BNSE, LS, or SM.
            sm_method (str, optional): Method of estimating SM kernels. Only valid with SM method.
            sm_opt (str, optional): Optimization method for SM kernels. Only valid with SM method.
            sm_maxiter (str, optional): Maximum iteration for SM kernels. Only valid with SM method.
            plot (bool): Show the PSD of the kernel after fitting SM kernels.
        """

        output_dims = self.dataset.get_output_dims()

        if not method.lower() in ['bnse', 'ls', 'sm']:
            raise ValueError("valid methods of estimation are BNSE, LS, and SM")

        if method.lower() == 'bnse':
            amplitudes, means, variances = self.dataset.get_bnse_estimation(self.Q)
        elif method.lower() == 'ls':
            amplitudes, means, variances = self.dataset.get_lombscargle_estimation(self.Q)
        else:
            amplitudes, means, variances = self.dataset.get_sm_estimation(self.Q, method=sm_method, optimizer=sm_opt, maxiter=sm_maxiter, plot=plot)
        if len(amplitudes) == 0:
            logger.warning('{} could not find peaks for MOSM'.format(method))
            return

        # input_dims must be the same for all channels (restriction of MOSM)
        constant = np.empty((output_dims, self.Q))
        for q in range(self.Q):
            constant[:,q] = np.array([amplitude[q,:] for amplitude in amplitudes]).mean(axis=0)
            self.model.kernel[q].variance.assign([variance[q,:] * 10.0 for variance in variances])

        for i, channel in enumerate(self.dataset):
            _, y = channel.get_train_data(transformed=True)
            constant[i,:] = constant[i,:] / constant[i,:].sum() * y.var()

        for q in range(self.Q):
            self.model.kernel[q].weight.assign(constant[:,q])

        noise = np.empty((output_dims,))
        for i, channel in enumerate(self.dataset):
            _, y = channel.get_train_data(transformed=True)
            noise[i] = y.var() / 30.0
        for q in range(self.Q):
            self.model.kernel[q].noise.assign(noise)
    
