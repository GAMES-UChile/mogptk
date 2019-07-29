import logging
from .data import Data
from .model import model
from .kernels import SpectralMixture, sm_init, SpectralMixtureOLD, MultiOutputSpectralMixture, SpectralMixtureLMC, ConvolutionalGaussian, CrossSpectralMixture
import numpy as np
import gpflow
import tensorflow as tf

def load(filename):
    graph = tf.Graph()
    session = tf.Session(graph=graph)
    with graph.as_default():
        with session.as_default():
            gpmodel = gpflow.saver.Saver().load(filename)

    model_type = gpmodel.mogptk_type
    name = gpmodel.mogptk_name
    data = Data._decode(gpmodel.mogptk_data)
    Q = gpmodel.mogptk_Q
    params = gpmodel.mogptk_params
    fixed_params = gpmodel.mogptk_fixed_params

    if model_type == 'SM':
        m = SM(data, Q, name)
    elif model_type == 'MOGP':
        m = MOGP(data, Q, name)
    elif model_type == 'CG':
        m = CG(data, Q, name)
    elif model_type == 'CSM':
        m = CSM(data, Q, name)
    elif model_type == 'SM_LMC':
        m = SM_LMC(data, Q, name)
    else:
        raise Exception("unknown model type '%s'" % (model_type))

    m.model = gpmodel
    m.params = params
    m.fixed_params = fixed_params
    m.graph = graph
    m.session = session
    return m

def transform_multioutput_data(x, y=None):
    chan = []
    if isinstance(x, list):
        for channel in range(len(x)):
            chan.append(channel * np.ones(len(x[channel])))
        chan = np.concatenate(chan)
    elif isinstance(x, dict):
        for channel, data in x.items():
            chan.append(channel * np.ones(len(data)))
        chan = np.concatenate(chan)
        x = np.array(list(x.values()))
    else:
        raise Exception("unknown data type for x")
    
    chan = chan.reshape(-1, 1)
    x = np.concatenate(x)
    x = np.concatenate((chan, x), axis=1)
    if y != None:
        y = np.concatenate(y).reshape(-1, 1)
    return x, y

def estimate_from_sm(data, Q):
    """
    Estimate kernel param with single ouput GP-SM

    Args:
        data (obj; mogptk.Data): Data class instance.

        Q (int): Number of components.

    returns: 
        params[q][name][output dim][input dim]
    """
    params = []
    for q in range(Q):
        params.append({'weight': [], 'mean': [], 'scale': []})

    for channel in range(data.get_output_dims()):
        sm_data = Data()
        sm_data.add(data.X[channel], data.Y[channel])

        sm = SM(sm_data, Q)
        sm.init_params()
        sm.train(method='BFGS', disp=True)

        for q in range(Q):
            params[q]['weight'].append(sm.params[q]['mixture_weights']),
            params[q]['mean'].append(sm.params[q]['mixture_means']),
            params[q]['scale'].append(sm.params[q]['mixture_scales']),

    for q in range(Q):
        params[q]['weight'] = np.array(params[q]['weight'])
        params[q]['mean'] = np.array(params[q]['mean']) * np.pi * 2
        params[q]['scale'] = np.array(params[q]['scale'])
    return params

class SM(model):
    """
    Single output GP with Spectral mixture kernel.
    """
    def __init__(self, data, Q=1, name="SM"):
        model.__init__(self, name, data, Q)

        input_dims = self.data.get_input_dims()
        output_dims = self.data.get_output_dims()
        if output_dims != 1:
            raise Exception("Single output Spectral Mixture kernel can only take one output dimension in the data")

        weights = np.abs(np.random.standard_normal((Q)))
        means = np.abs(np.random.standard_normal((Q, input_dims)))
        scales = np.abs(np.random.standard_normal((Q, input_dims)))

        for q in range(Q):
            self.params.append({
                'mixture_weights': weights[q],
                'mixture_means': np.array(means[q]),
                'mixture_scales': np.array(scales[q]),
            })

    def init_params(self, method='BNSE'):
        """
        Initialize parameters of kernel from data using different methods.

        Kernel parameters can be initialized using 3 heuristics using the train data:

        -'random' is taking the inverse of lengthscales drawn from truncated Gaussian
            N(0, max_dist^2), the means drawn from
            Unif(0, 0.5 / minimum distance between two points),
            and the mixture weights by taking the stdv of the y values divided by the
            number of mixtures.

        -'LS' is using Lomb Scargle periodogram for estimating the PSD,
            and using the first Q peaks as the means and mixture weights.

        -'BNSE' third is using the BNSE (Tobar 2018) to estimate the PSD 
            and use the first Q peaks as the means and mixture weights.

        *** Only for single input dimension for each channel.
        """

        if method not in ['random', 'LS', 'BNSE']:
            raise Exception("Posible methods are 'random', 'LS' and 'BNSE' (see documentation).")

        if method=='random':
            x, y = self._transform_data(self.data.X, self.data.Y)
            weights, means, scales = sm_init(x, y, self.Q)

            for q in range(self.Q):
                self.params[q]['mixture_weights'] = weights[q]
                self.params[q]['mixture_means'] = np.array(means[q])
                self.params[q]['mixture_scales'] = np.array(scales[q])

        elif method=='LS':
            pass

        elif method=='BNSE':
            means, amplitudes = self.data.get_bnse_estimation(self.Q)
            if amplitudes[0].shape[1] != self.Q or means[0].shape[1] != self.Q:
                logging.warning('BNSE could not find peaks for SM')
                return

            weights = amplitudes[0] * self.data.Y[0].std()
            weights = np.sqrt(weights/np.sum(weights))
            for q in range(self.Q):
                self.params[q]['mixture_weights'] = weights[0][q]
                self.params[q]['mixture_means'] = means[0].T[q] / np.pi / 2.0

    def _transform_data(self, x, y=None):
        if isinstance(x, dict):
            x = np.array(list(x.values()))
        elif isinstance(x, list):
            x = np.array(x)
        x = np.squeeze(x, axis=0) # not using multi output dims
        y = np.array(y)
        return x, y.T

    def _kernel(self):
        weights = np.array([self.params[q]['mixture_weights'] for q in range(self.Q)])
        means = np.array([self.params[q]['mixture_means'] for q in range(self.Q)])
        scales = np.array([self.params[q]['mixture_scales'] for q in range(self.Q)]).T
        return SpectralMixture(
            self.Q,
            weights,
            means,
            scales,
            self.data.get_input_dims(),
        )

    def _update_params(self, trainables):
        for key, val in trainables.items():
            names = key.split("/")
            if len(names) == 3 and names[1] == 'kern':
                name = names[2]
                if name == 'mixture_scales':
                    val = val.T
                for q in range(len(val)):
                    self.params[q][name] = val[q]

class MOSM(model):
    """
    Multi Output Spectral Mixture kernel as proposed by our paper.

    It takes a number of components Q and allows for recommended initial
    parameter estimation to improve optimization outputs.
    """
    def __init__(self, data, Q=1, name="MOSM"):
        model.__init__(self, name, data, Q)

        input_dims = self.data.get_input_dims()
        output_dims = self.data.get_output_dims()
        for _ in range(Q):
            self.params.append({
                "magnitude": np.random.standard_normal((output_dims)),
                "mean": np.random.standard_normal((input_dims, output_dims)),
                "variance": np.random.random((input_dims, output_dims)),
                "delay": np.zeros((input_dims, output_dims)),
                "phase": np.zeros((output_dims)),
                "noise": np.random.random((output_dims)),
            })

    def init_means(self):
        """
        Initialize spectral means using BNSE[1]
        """
        peaks, _ = self.data.get_bnse_estimation(self.Q)
        for q in range(self.Q):
            self.params[q]["mean"] = peaks[0].T[q].reshape(-1, 1)
            for channel in range(1,self.data.get_output_dims()):
                self.params[q]["mean"] = np.append(self.params[q]["mean"], peaks[channel].T[q].reshape(-1, 1))

    def init_params(self, mode='full'):
        """
        Initialize kernel parameters, spectral mean, (and optionaly) variance and mixture weights. 

        The initialization is done fitting a single output GP with Sepectral mixture (SM)
        kernel for each channel. Furthermore, each GP-SM in fitted initializing
        its parameters with Bayesian Nonparametric Spectral Estimation (BNSE)

        Args:
            mode (str): Parameters to initialize, 'means' estimates only the means trough BNSE
                directly. 'full' estimates the spectral mean, variance and weights with GP-SM.
        """

        if mode not in ['full', 'means']:
            raise Exception("Posible modes are either 'full' or 'means'.")

        if mode=='means':
            self.init_means()

        elif mode=='full':
            params = estimate_from_sm(self.data, self.Q)
            for q in range(self.Q):
                self.params[q]["magnitude"] = params[q]['weight']
                self.params[q]["mean"] = params[q]['mean'].T
                self.params[q]["variance"] = params[q]['scale'].T

    def _transform_data(self, x, y=None):
        return transform_multioutput_data(x, y)

    def _kernel(self):
        for q in range(self.Q):
            kernel = MultiOutputSpectralMixture(
                self.data.get_input_dims(),
                self.data.get_output_dims(),
                self.params[q]["magnitude"],
                self.params[q]["mean"],
                self.params[q]["variance"],
                self.params[q]["delay"],
                self.params[q]["phase"],
                self.params[q]["noise"],
            )

            if q == 0:
                kernel_set = kernel
            else:
                kernel_set += kernel
        return kernel_set

class CSM(model):
    """
    Cross Spectral Mixture kernel with Q components and Rq latent functions
    (TODO: true?).
    """
    def __init__(self, data, Q=1, Rq=1, name="CSM"):
        model.__init__(self, name, data, Q)

        if Rq != 1:
            raise Exception("Rq != 1 is not (yet) supported") # TODO: support
        self.Rq = Rq
        
        input_dims = self.data.get_input_dims()
        output_dims = self.data.get_output_dims()
        for _ in range(Q):
            self.params.append({
                "constant": np.random.random((Rq, output_dims)),
                "mean": np.random.random((input_dims)),
                "variance": np.random.random((input_dims)),
                "phase": np.zeros((Rq, output_dims)),
            })
    
    def init_params(self):
        """
        Initialize kernel parameters, spectral mean, (and optionaly) variance and mixture weights. 

        The initialization is done fitting a single output GP with Sepectral mixture (SM)
        kernel for each channel. Furthermore, each GP-SM in fitted initializing
        its parameters with Bayesian Nonparametric Spectral Estimation (BNSE)
        """
        data = self.data.copy()
        data.normalize()
        all_params = estimate_from_sm(data, self.Q)
        print(all_params)

        params = {'weight': [], 'mean': [], 'scale': []}
        for channel in range(len(all_params)):
            params['weight'] = np.append(params['weight'], all_params[channel]['weight'])
            params['mean'] = np.append(params['mean'], all_params[channel]['mean'])
            params['scale'] = np.append(params['scale'], all_params[channel]['scale'])

        indices = np.argsort(params['weight'])[::-1]
        for q in range(self.Q):
            if q < len(indices):
                i = indices[q]
                self.params[q]['mean'] = np.array([params['mean'][i]])
                self.params[q]['variance'] = np.array([params['scale'][i]])

    def _transform_data(self, x, y=None):
        return transform_multioutput_data(x, y)

    def _kernel(self):
        for q in range(self.Q):
            kernel = CrossSpectralMixture(
                self.data.get_input_dims(),
                self.data.get_output_dims(),
                self.Rq,
                self.params[q]["constant"],
                self.params[q]["mean"],
                self.params[q]["variance"],
                self.params[q]["phase"],
            )

            if q == 0:
                kernel_set = kernel
            else:
                kernel_set += kernel
        return kernel_set

class SM_LMC(model):
    """
    Spectral Mixture - Linear Model of Coregionalization kernel with Q components and Rq latent functions
    (TODO: true?)."""
    def __init__(self, data, Q=1, Rq=1, name="SM-LMC"):
        model.__init__(self, name, data, Q)

        if Rq != 1:
            raise Exception("Rq != 1 is not (yet) supported") # TODO: support
        self.Rq = Rq
        
        input_dims = self.data.get_input_dims()
        output_dims = self.data.get_output_dims()
        for _ in range(Q):
            self.params.append({
                "constant": np.random.standard_normal((Rq, output_dims)),
                "mean": np.random.random((input_dims)),
                "variance": np.random.random((input_dims)),
            })
    
    def init_params(self):
        """
        Initialize kernel parameters, spectral mean, (and optionaly) variance and mixture weights. 

        The initialization is done fitting a single output GP with Sepectral mixture (SM)
        kernel for each channel. Furthermore, each GP-SM in fitted initializing
        its parameters with Bayesian Nonparametric Spectral Estimation (BNSE)
        """
        data = self.data.copy()
        data.normalize()
        all_params = estimate_from_sm(data, self.Q)
        print(all_params)

        params = {'weight': [], 'mean': [], 'scale': []}
        for channel in range(len(all_params)):
            params['weight'] = np.append(params['weight'], all_params[channel]['weight'])
            params['mean'] = np.append(params['mean'], all_params[channel]['mean'])
            params['scale'] = np.append(params['scale'], all_params[channel]['scale'])

        indices = np.argsort(params['weight'])[::-1]
        for q in range(self.Q):
            if q < len(indices):
                i = indices[q]
                self.params[q]['mean'] = np.array([params['mean'][i]])
                self.params[q]['variance'] = np.array([params['scale'][i]])

    def _transform_data(self, x, y=None):
        return transform_multioutput_data(x, y)

    def _kernel(self):
        for q in range(self.Q):
            kernel = SpectralMixtureLMC(
                self.data.get_input_dims(),
                self.data.get_output_dims(),
                self.Rq,
                self.params[q]["constant"],
                self.params[q]["mean"],
                self.params[q]["variance"],
            )

            if q == 0:
                kernel_set = kernel
            else:
                kernel_set += kernel
        return kernel_set

class CG(model):
    """
    CG is the Convolutional Gaussian kernel with Q components.
    """
    def __init__(self, data, Q=1, name="CG"):
        model.__init__(self, name, data, Q)
        
        input_dims = self.data.get_input_dims()
        output_dims = self.data.get_output_dims()
        for _ in range(Q):
            self.params.append({
                "constant": np.random.random((output_dims)),
                "variance": np.random.random((input_dims, output_dims)),
            })

    def init_params(self):
        """
        Initialize kernel parameters, variance and mixture weights. 

        The initialization is done fitting a single output GP with Sepectral mixture (SM)
        kernel for each channel. Furthermore, each GP-SM in fitted initializing
        its parameters with Bayesian Nonparametric Spectral Estimation (BNSE)
        """
        params = estimate_from_sm(self.data, self.Q) # TODO: fix spectral mean
        for q in range(self.Q):
            self.params[q]["variance"] = params[q]['scale'].T

    def _transform_data(self, x, y=None):
        return transform_multioutput_data(x, y)

    def _kernel(self):
        for q in range(self.Q):
            kernel = ConvolutionalGaussian(
                self.data.get_input_dims(),
                self.data.get_output_dims(),
                self.params[q]["constant"],
                self.params[q]["variance"],
            )

            if q == 0:
                kernel_set = kernel
            else:
                kernel_set += kernel
        return kernel_set
    
