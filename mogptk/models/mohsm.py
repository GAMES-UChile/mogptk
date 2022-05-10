import torch
import numpy as np

from ..dataset import DataSet
from ..model import Model, Exact, logger
from ..gpr import MultiOutputHarmonizableSpectralKernel,MixtureKernel


class MOHSM(Model):
    """
    Multi-output harmonizable spectral kernel with `P` components and `Q` subcomponents [1]. The parameters will be randomly instantiated, use `init_parameters()` to initialize the parameters to reasonable values for the current data set.

    Args:
        dataset (mogptk.dataset.DataSet): `DataSet` object of data for all channels.
        P (int): Number of components.
        Q (int): Number of subcomponents.
        model: Gaussian process model to use, such as `mogptk.model.Exact`.
        mean (mogptk.gpr.mean.Mean): The mean class.
        name (str): Name of the model.
        rescale_x (bool): Rescale the X axis to [0,1000] to help training.

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
    >>> model = mogptk.MOHSM(dataset, P=2, Q=2)
    >>> model.init_parameters()
    >>> model.train()
    >>> model.predict()
    >>> dataset.plot()

    [1] M. Altamirano, "Nonstationary Multi-Output Gaussian Processes via Harmonizable Spectral Mixtures, 2021
    """
    def __init__(self, dataset, P=1, Q=1, model=Exact(), mean=None, name="MOHSM", rescale_x=False):
        if not isinstance(dataset, DataSet):
            dataset = DataSet(dataset)

        spectral = MultiOutputHarmonizableSpectralKernel(
            output_dims=dataset.get_output_dims(),
            input_dims=dataset.get_input_dims()[0],
        )
        kernel_p = MixtureKernel(spectral, Q)
        
        kernel = MixtureKernel(kernel_p, P)
        
        super(MOHSM, self).__init__(dataset, kernel, model, mean, name, rescale_x)
        self.Q = Q
        self.P = P
        self.rescale_x = rescale_x

    
    def init_parameters(self, method='BNSE', sm_init='SM', sm_method='Adam', sm_iters=100, sm_params={},sm_plot=False):
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


        #TODO
        if not self.rescale_x:
            raise ValueError("method valid only for x rescaled ")
        
        for p in range(self.P):
            
            for q in range(self.Q):
                if self.P!=1:
                    self.gpr.kernel[p*self.Q+q].center.assign((1000*p/(self.P-1))*np.ones(input_dims[0]))
                    self.gpr.kernel[p*self.Q+q].lengthscale.assign(((self.P+1)/1000)*np.ones(output_dims))

            dataset = self.dataset.copy()
            
            if input_dims[0]==1:
                for i, channel in enumerate(dataset):
                    m = len(dataset[i].X[0])-1
                    if self.P ==2:
                        if p == 0: 
                            channel.filter(dataset[i].X[0][0], dataset[i].X[0][int(m/2)])
                        if p == 1: 
                            channel.filter(dataset[i].X[0][int(m/2)], dataset[i].X[0][m])
                    else:

                        if p == 0:
                            channel.filter(dataset[i].X[0][0], dataset[i].X[0][int(m/(self.P+1))])
                        elif p == self.P-1:
                            channel.filter(dataset[i].X[0][int(m*(p+1)/(self.P+1))], dataset[i].X[0][int(m)])
                        else:
                            channel.filter(dataset[i].X[0][int(m*p/(self.P+1))], dataset[i].X[0][int(m*(p+2)/(self.P+1))])

            if method.lower() == 'bnse':
                amplitudes, means, variances = dataset.get_bnse_estimation(self.Q)
            elif method.lower() == 'ls':
                amplitudes, means, variances = dataset.get_lombscargle_estimation(self.Q)
            else:
                amplitudes, means, variances = dataset.get_sm_estimation(self.Q, method=sm_init, optimizer=sm_method, iters=sm_iters, params=sm_params, plot=sm_plot)
            if len(amplitudes) == 0:
                logger.warning('{} could not find peaks for MOHSM'.format(method))
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
                self.gpr.kernel[p*self.Q+q].mean.assign(mean)
                self.gpr.kernel[p*self.Q+q].variance.assign(variance)

            # normalize proportional to channels variances
            for j, channel in enumerate(dataset):
                _, y = channel.get_train_data(transformed=self.rescale_x)
                x, _ = channel.get_data(transformed=self.rescale_x)
                if 0.0 < magnitude[j,:].sum():
                    magnitude[j,:] = (np.sqrt(magnitude[j,:] / magnitude[j,:].sum() * y.var())) * 2

            for q in range(self.Q):
                self.gpr.kernel[p*self.Q+q].magnitude.assign(magnitude[:,q]/np.sqrt(self.gpr.kernel[p*self.Q+q].lengthscale().cpu().detach().numpy()))

            # TODO
            # noise = np.empty((output_dims,))
            # for j, channel in enumerate(dataset):
            #     _, y = channel.get_train_data(transformed=self.rescale_x)
            #     noise[j] = y.var() / 30.0
            # for q in range(self.Q):
            #     self.gpr.kernel[p*self.Q+q].noise.assign(noise)
