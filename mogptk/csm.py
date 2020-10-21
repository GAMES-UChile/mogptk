import numpy as np
from .model import model
from .kernels import CrossSpectralKernel, MixtureKernel
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
    def __init__(self, dataset, Q=1, Rq=1, name="CSM"):
        if Rq != 1:
            raise Exception("Rq != 1 is not (yet) supported") # TODO: support

        model.__init__(self, name, dataset)
        self.Q = Q
        self.Rq = Rq

        spectral = CrossSpectralKernel(
            output_dims=self.dataset.get_output_dims(),
            input_dims=self.dataset.get_input_dims()[0],
            Rq=Rq,
        )
        kernel = MixtureKernel(spectral, Q=Q)
        self._build(kernel)
        self.model.noise.assign(0.0, lower=0.0, trainable=False)  # handled by MultiOutputKernel
    
    def init_parameters(self, method='BNSE', sm_method='BNSE', sm_opt='LBFGS', sm_maxiter=3000, plot=False):
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
            method (str, optional): Method of estimation, such as BNSE, LS, GMM, or SM.
            sm_method (str, optional): Method of estimating SM kernels. Only valid with SM method.
            sm_opt (str, optional): Optimization method for SM kernels. Only valid with SM method.
            sm_maxiter (str, optional): Maximum iteration for SM kernels. Only valid with SM method.
            plot (bool, optional): Show the PSD of the kernel after fitting SM kernels. Only valid in 'SM' mode.
        """

        output_dims = self.dataset.get_output_dims()

        if method.lower() in ['bnse', 'ls', 'gmm']:
            if method.lower() == 'bnse':
                amplitudes, means, variances = self.dataset.get_bnse_estimation(self.Q)
            elif method.lower() == 'ls':
                amplitudes, means, variances = self.dataset.get_lombscargle_estimation(self.Q)
            else:
                amplitudes, means, variances = self.dataset.get_gmm_estimation(self.Q)
            if len(amplitudes) == 0:
                logger.warning('{} could not find peaks for CSM'.format(method))
                return

            constant = np.empty((output_dims, self.Q, self.Rq))
            for q in range(self.Q):
                for i in range(len(self.dataset)):
                    constant[i,q,:] = amplitudes[i][:,q].mean()
                mean = np.array(means)[:,:,q].mean(axis=0)
                variance = np.array(variances)[:,:,q].mean(axis=0)
                self.model.kernel[q].mean.assign(mean)
                self.model.kernel[q].variance.assign(variance * 5.0)

            # normalize proportional to channel variance
            for i, channel in enumerate(self.dataset):
                _, y = channel.get_train_data(transformed=True)
                constant[i,:,:] = constant[i,:,:] / constant[i,:,:].sum() * y.var() * 2

            for q in range(self.Q):
                self.model.kernel[q].amplitude.assign(constant[:,q,:])

        elif method.lower() == 'sm':
            params = _estimate_from_sm(self.dataset, self.Q, method=sm_method, optimizer=sm_opt, maxiter=sm_maxiter, plot=plot)

            constant = np.empty((output_dims, self.Q, self.Rq))
            for q in range(self.Q):
                constant[:,q,:] = params[q]['weight'].mean(axis=0)

            # normalize proportional to channel variance
            for i, channel in enumerate(self.dataset):
                _, y = channel.get_train_data(transformed=True)
                constant[i,:,:] = constant[i,:,:] / constant[i,:,:].sum() * y.var() * 2
                
            for q in range(self.Q):
                self.model.kernel[q].amplitude.assign(constant[:,q,:])
                self.model.kernel[q].mean.assign(params[q]['mean'].mean(axis=1))
                self.model.kernel[q].variance.assign(params[q]['scale'].mean(axis=1) * 2)
        else:
            raise ValueError("valid methods of estimation are BNSE, LS, GMM, and SM")

        #noise = np.empty((output_dims,))
        #for i, channel in enumerate(self.dataset):
        #    _, y = channel.get_train_data(transformed=True)
        #    noise[i] = y.var() / 30
        #self.set_parameter(self.Q, 'noise', noise)
