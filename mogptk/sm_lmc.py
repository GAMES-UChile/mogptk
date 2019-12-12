from .model import model
from .kernels import SpectralMixtureLMC, Noise

class SM_LMC(model):
    """
    Spectral Mixture - Linear Model of Coregionalization kernel with Q components and Rq latent functions.
        
    Args:
        dataset (DataSet): DataSet object of data for all channels.
        Q (int): Number of components to use.
        Rq (int): Number of subcomponents to use.
        name (string): Name of the model.
        likelihood (gpflow.likelihoods): Likelihood to use from GPFlow, if None a default exact inference Gaussian likelihood is used.
        variational (bool): If True, use variational inference to approximate function values as Gaussian. If False it will use Monte Carlo Markov Chain (default).
        sparse (bool): If True, will use sparse GP regression. Defaults to False.
        like_params (dict): Parameters to GPflow likelihood.
    """
    def __init__(self, dataset, Q=1, Rq=1, name="SM-LMC", likelihood=None, variational=False, sparse=False, like_params={}):
        if Rq != 1:
            raise Exception("Rq != 1 is not (yet) supported") # TODO: support
        self.Rq = Rq

        for q in range(self.Q):
            kernel = SpectralMixtureLMC(
                self.get_input_dims(),
                self.get_output_dims(),
                self.Rq,
            )
            if q == 0:
                kernel_set = kernel
            else:
                kernel_set += kernel
        kernel_set += Noise(self.get_input_dims(), self.get_output_dims())

        model.__init__(self, name, dataset, kernel_set, likelihood, variational, sparse, like_params)
    
    def estimate_params(self, method='BNSE', sm_init='BNSE', sm_method='BFGS', sm_maxiter=2000, plot=False):
        """
        Estimate kernel parameters.

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
        
        self.params[self.Q]['noise'] = _estimate_noise_var(self.data)
        
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
