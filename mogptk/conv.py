import numpy as np
from .model import model
from .kernels import ConvolutionalGaussian, Noise
from .sm import _estimate_from_sm

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
    def __init__(self, dataset, Q=1, name="CONV", likelihood=None, variational=False, sparse=False, like_params={}):
        if len(dataset)<2:
            raise Exception("Number of channels equal 1, model CONV must be used with at least 2, for 1 channel use SM instead.")
        model.__init__(self, name, dataset)
        self.Q = Q

        for q in range(self.Q):
            kernel = ConvolutionalGaussian(
                self.dataset.get_input_dims()[0],
                self.dataset.get_output_dims(),
            )
            if q == 0:
                kernel_set = kernel
            else:
                kernel_set += kernel
        kernel_set += Noise(self.dataset.get_input_dims()[0], self.dataset.get_output_dims())
        self._build(kernel_set, likelihood, variational, sparse, like_params)

    def estimate_parameters(self, method='SM', sm_method='random', sm_opt='BFGS', sm_maxiter=2000, plot=False):
        """
        Estimate kernel parameters, variance and mixture weights. 

        The initialization is done fitting a single output GP with Sepectral mixture (SM)
        kernel for each channel with spectral means fixed to 0 for all Q.

        In all cases the noise is initialized with 1/30 of the variance 
        for each channel.

        Args:
            method (str, optional): Method of estimation, possible value is 'SM'.
            sm_method (str, optional): Method of estimating SM kernels. Only valid in 'SM' mode.
            sm_opt (str, optional): Optimization method for SM kernels. Only valid in 'SM' mode.
            sm_maxiter (str, optional): Maximum iteration for SM kernels. Only valid in 'SM' mode.
            plot (bool): Show the PSD of the kernel after fitting SM kernels.
        """
        if method == 'SM':
            params = _estimate_from_sm(self.dataset, self.Q, method=sm_method, optimizer=sm_opt, maxiter=sm_maxiter, plot=plot, fix_means=True)

            constant = np.empty((self.Q, self.dataset.get_output_dims()))
            for q in range(self.Q):
                constanst = params[q]['weight'].mean(axis=0)
                self.set_parameter(q, 'variance', params[q]['scale'] * 10)

            for channel, data in enumerate(self.dataset):
                constant[:, channel] = constant[:, channel] / constant[:, channel].sum() * data.Y[data.mask].var()

            for q in range(self.Q):
                self.set_parameter(q, 'constant', constant[q, :])
        else:
            raise Exception("possible methods of estimation are only 'SM'")

        noise = np.empty((self.dataset.get_output_dims()))
        for i, channel in enumerate(self.dataset):
            noise[i] = (channel.Y).var() / 30
        self.set_parameter(self.Q, 'noise', noise)
    
