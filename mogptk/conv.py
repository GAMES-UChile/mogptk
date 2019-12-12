from .model import model
from .kernels import ConvolutionalGaussian, Noise

class CONV(model):
    """
    CONV is the Convolutional Gaussian kernel with Q components.
        
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
    def __init__(self, dataset, Q=1, name="CG", likelihood=None, variational=False, sparse=False, like_params={}):
        for q in range(self.Q):
            kernel = ConvolutionalGaussianOLD(
                self.get_input_dims(),
                self.get_output_dims(),
            )
            if q == 0:
                kernel_set = kernel
            else:
                kernel_set += kernel
        kernel_set += Noise(self.get_input_dims(), self.get_output_dims())
        model.__init__(self, name, dataset, kernel_set, likelihood, variational, sparse, like_params)

    def estimate_params(self, sm_init='random', sm_method='BFGS', sm_maxiter=2000, plot=False):
        """
        Estimate kernel parameters, variance and mixture weights. 

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
    
