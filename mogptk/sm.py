from .model import model
from .kernels import SpectralMixture, sm_init, Noise

def _estimate_from_sm(data, Q, init='BNSE', method='BFGS', maxiter=2000, plot=False, fix_means=False):
    """
    Estimate kernel param with single ouput GP-SM

    Args:
        data (obj; mogptk.Data): Data class instance.

        Q (int): Number of components.
        init (str): Method to initialize, 'BNSE', 'LS' or 'Random'
        method (str): Optimization method, either 'Adam' or any
            scipy optimizer.
        maxiter (int): Maximum number of iteration.
        plot (bool): If true plot the kernel PSD of each channel.
        fix_means(bool): Fix spectral means to zero in trainning.

    returns: 
        params[q][name][output dim][input dim]
    """
    input_dims = data[0].get_input_dims()
    output_dims = len(data)

    params = []
    for q in range(Q):
        params.append({
            'weight': np.empty((input_dims, output_dims)),
            'mean': np.empty((input_dims, output_dims)),
            'scale': np.empty((input_dims, output_dims)),
        })

    for channel in range(output_dims):
        for i in range(input_dims):  # TODO one SM per channel
            sm = SM(data[channel], Q)
            sm.init_params(init)

            if fix_means:
                for q in range(sm.Q):
                    sm.params[q]['mixture_means'] = np.zeros(input_dims)
                    sm.params[q]['mixture_scales'] *= 100

            sm.build()

            if fix_means:
                sm.fix_param('mixture_means')

            sm.train(method=method, maxiter=maxiter, tol=1e-50)

            if plot:
                nyquist = data[channel].get_nyquist_estimation()
                means = sm._get_param_across('mixture_means')
                weights = sm._get_param_across('mixture_weights')
                scales = sm._get_param_across('mixture_scales')
                plot_spectrum(means, scales, weights=weights, nyquist=nyquist, title=data[channel].name)

            for q in range(Q):
                params[q]['weight'][i,channel] = sm.params[q]['mixture_weights']
                params[q]['mean'][i,channel] = sm.params[q]['mixture_means'] * np.pi * 2
                params[q]['scale'][i,channel] = sm.params[q]['mixture_scales']

    return params

class SM(model):
    """
    Single output GP with Spectral mixture kernel.
        
    Args:
        data (Data,list of Data): Data object or list of Data objects for each channel. Only one channel allowed for SM.
        Q (int): Number of components to use.
        name (string): Name of the model.
        likelihood (gpflow.likelihoods): Likelihood to use from GPFlow, if None a default exact inference Gaussian likelihood is used.
        variational (bool): If True, use variational inference to approximate function values as Gaussian. If False it will use Monte Carlo Markov Chain (default).
        sparse (bool): If True, will use sparse GP regression. Defaults to False.
        like_params (dict): Parameters to GPflow likelihood.
    """
    def __init__(self, data, Q=1, name="SM", likelihood=None, variational=False, sparse=False, like_params={}):
        input_dims = self.get_input_dims()
        output_dims = self.get_output_dims()
        if output_dims != 1:
            raise Exception("single output Spectral Mixture kernel can only take one output dimension in the data")

        #weights = np.array([self.params[q]['mixture_weights'] for q in range(self.Q)])
        #means = np.array([self.params[q]['mixture_means'] for q in range(self.Q)])
        #scales = np.array([self.params[q]['mixture_scales'] for q in range(self.Q)]).T
        kernel = SpectralMixture(
            input_dims,
            self.Q,
        )
        # TODO: get x, y from dataset => all kernels must accept data of the same format
        model.__init__(self, data, name, kernel, likelihood, variational, sparse, like_params)

    def _transform_data(self, x, y=None):
        if y == None:
            return x[0]
        return x[0], np.expand_dims(y[0], 1)

    def init_params(self, method='BNSE'):
        """
        Initialize parameters of kernel from data using different methods.

        Kernel parameters can be initialized using 3 heuristics using the train data:

        'random': (Taken from phd thesis from Andrew wilson 2014) is taking the inverse
            of lengthscales drawn from truncated Gaussian N(0, max_dist^2), the
            means drawn from Unif(0, 0.5 / minimum distance between two points),
            and the mixture weights by taking the stdv of the y values divided by the
            number of mixtures.
        'LS'_ is using Lomb Scargle periodogram for estimating the PSD,
            and using the first Q peaks as the means and mixture weights.
        'BNSE': third is using the BNSE (Tobar 2018) to estimate the PSD 
            and use the first Q peaks as the means and mixture weights.

        In all cases the noise is initialized with 1/30 of the variance of each channel.

        *** Only for single input dimension for each channel.
        """

        if method not in ['random', 'LS', 'BNSE']:
            raise Exception("posible methods are 'random', 'LS' and 'BNSE' (see documentation).")

        if method=='random':
            x, y = self.data[0].get_obs()
            weights, means, scales = sm_init(x, y, self.Q)
            for q in range(self.Q):
                self.params[q]['mixture_weights'] = weights[q]
                self.params[q]['mixture_means'] = np.array(means[q])
                self.params[q]['mixture_scales'] = np.array(scales[q])

        elif method=='LS':
            amplitudes, means, variances = self.data[0].get_ls_estimation(self.Q)
            if len(amplitudes) == 0:
                logging.warning('BNSE could not find peaks for SM')
                return

            for q in range(self.Q):
                self.params[q]['mixture_weights'] = amplitudes[:, q].mean() / amplitudes.mean()
                self.params[q]['mixture_means'] = means.T[q]
                self.params[q]['mixture_scales'] = variances.T[q] * 2

        elif method=='BNSE':
            amplitudes, means, variances = self.data[0].get_bnse_estimation(self.Q)
            if np.sum(amplitudes) == 0.0:
                logging.warning('BNSE could not find peaks for SM')
                return

            for q in range(self.Q):
                self.params[q]['mixture_weights'] = amplitudes[:, q].mean() / amplitudes.mean()
                self.params[q]['mixture_means'] = means.T[q]
                self.params[q]['mixture_scales'] = variances.T[q] * 2

    def _update_params(self, trainables):
        for key, val in trainables.items():
            names = key.split("/")
            if len(names) == 3 and names[1] == 'kern':
                name = names[2]
                if name == 'mixture_scales':
                    val = val.T
                for q in range(len(val)):
                    self.params[q][name] = val[q]


    def plot_psd(self, figsize=(10, 4), title='', log_scale=False):
        """
        Plot power spectral density of 
        single output GP-SM
        """
        means = self._get_param_across('mixture_means').reshape(-1)
        weights = self._get_param_across('mixture_weights').reshape(-1)
        scales = self._get_param_across('mixture_scales').reshape(-1)
        
        # calculate bounds
        x_low = norm.ppf(0.001, loc=means, scale=scales).min()
        x_high = norm.ppf(0.99, loc=means, scale=scales).max()
        
        x = np.linspace(x_low, x_high + 1, 1000)
        psd = np.zeros_like(x)

        f, ax = plt.subplots(1, 1, figsize=figsize)
        # fig, axes = plt.subplots(1, 1, figsize=(20, 5)
        
        for q in range(self.Q):
            single_psd = weights[q] * norm.pdf(x, loc=means[q], scale=scales[q])
            ax.plot(x, single_psd, '--', lw=1.2, c='xkcd:strawberry', zorder=2)
            ax.axvline(means[q], ymin=0.001, ymax=0.1, lw=2, color='grey')
            psd = psd + single_psd
            
        # symmetrize PSD
        if psd[x<0].size != 0:
            psd = psd + np.r_[psd[x<0][::-1], np.zeros((x>=0).sum())]
            
        ax.plot(x, psd, lw=2.5, c='r', alpha=0.7, zorder=1)
        ax.set_xlim(0, x[-1] + 0.1)
        if log_scale:
            ax.set_yscale('log')
        ax.set_xlabel(r'$\omega$')
        ax.set_ylabel('PSD')
        ax.set_title(title)
