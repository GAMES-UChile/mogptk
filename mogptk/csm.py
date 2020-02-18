import numpy as np
from .model import model
from .kernels import CrossSpectralMixture, Noise
from .sm import _estimate_from_sm

class CSM(model):
    """
    Cross Spectral Mixture kernel [1] with Q components and Rq latent functions.

    The model contain the dataset and the associated gpflow model, 
    when the mogptk.Model is instanciated the gpflow model is built 
    using random parameters.

    Args:
        dataset (mogptk.dataset.DataSet): DataSet object of data for all channels.
        Q (int, optional): Number of components.
        Rq (int, optional): Sub components por components.
        name (str, optional): Name of the model.
        likelihood (gpflow.likelihoods, optional): Likelihood to use from GPFlow, if None a default exact inference Gaussian likelihood is used.
        variational (bool, optional): If True, use variational inference to approximate function values as Gaussian. If False it will use Monte Carlo Markov Chain.
        sparse (bool, optional): If True, will use sparse GP regression.
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
    >>> model = mogptk.CSM(data_list, Q=2)
    >>> model.build()
    >>> model.train()
    >>> model.plot_prediction()

    [1] K.R. Ulrich et al, "GP Kernels for Cross-Spectrum Analysis", Advances in Neural Information Processing Systems 28, 2015
    """
    def __init__(self, dataset, Q=1, Rq=1, name="CSM", likelihood=None, variational=False, sparse=False, like_params={}):
        if Rq != 1:
            raise Exception("Rq != 1 is not (yet) supported") # TODO: support

        model.__init__(self, name, dataset)
        self.Q = Q
        self.Rq = Rq

        for q in range(self.Q):
            kernel = CrossSpectralMixture(
                self.dataset.get_input_dims()[0],
                self.dataset.get_output_dims(),
                self.Rq,
            )
            if q == 0:
                kernel_set = kernel
            else:
                kernel_set += kernel
        kernel_set += Noise(self.dataset.get_input_dims()[0], self.dataset.get_output_dims())
        self._build(kernel_set, likelihood, variational, sparse, like_params)
    
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

        if method == 'BNSE':
            amplitudes, means, variances = self.dataset.get_bnse_estimation(self.Q)
            if len(amplitudes) == 0:
                logger.warning('BNSE could not find peaks for CSM')
                return

            constant = np.empty((self.Q, self.Rq, self.dataset.get_output_dims()))
            for q in range(self.Q):
                for channel in range(len(self.dataset)):
                    constant[q, :,channel] = amplitudes[channel][:,q].mean()
            
                mean = np.array(means)[:,:,q].mean(axis=0)
                variance = np.array(variances)[:,:,q].mean(axis=0)

                self.set_parameter(q, 'mean', mean * 2 * np.pi)
                self.set_parameter(q, 'variance', variance * 5)

            # normalize proportional to channel variance
            for channel, data in enumerate(self.dataset):
                constant[:, :, channel] = constant[:, :, channel] / constant[:, :, channel].sum() * data.Y[data.mask].var() * 2

            for q in range(self.Q):
                self.set_parameter(q, 'constant', constant[q, :, :])

        elif method == 'SM':
            params = _estimate_from_sm(self.dataset, self.Q, method=sm_method, optimizer=sm_opt, maxiter=sm_maxiter, plot=plot)
            constant = np.empty((self.Q, self.Rq, self.dataset.get_output_dims()))

            for q in range(self.Q):
                constant[q, :, :] = params[q]['weight'].mean(axis=0).reshape(self.Rq, -1)
                self.set_parameter(q, 'mean', params[q]['mean'].mean(axis=1))
                self.set_parameter(q, 'variance', params[q]['scale'].mean(axis=1) * 2)

            # normalize proportional to channel variance
            for channel, data in enumerate(self.dataset):
                constant[:, :, channel] = constant[:, :, channel] / constant[:, :, channel].sum() * data.Y[data.mask].var() * 2
                
            for q in range(self.Q):
                self.set_parameter(q, 'constant', constant[q, :, :])
        else:
            raise Exception("possible methods of estimation are either 'BNSE' or 'SM'")

        noise = np.empty((self.dataset.get_output_dims()))
        for i, channel in enumerate(self.dataset):
            noise[i] = (channel.Y).var() / 30
        self.set_parameter(self.Q, 'noise', noise)
