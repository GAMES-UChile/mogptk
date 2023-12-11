import sys
import torch
import numpy as np
from scipy.stats import qmc, gaussian_kde
from IPython.display import display, HTML
from . import Parameter, Mean, Kernel, MultiOutputKernel, Likelihood, MultiOutputLikelihood, GaussianLikelihood, config, plot_gram

import warnings
warnings.simplefilter('ignore', torch.jit.TracerWarning)

def _init_grid(N, X):
    n = np.power(N,1.0/X.shape[1])
    if not n.is_integer():
        raise ValueError("number of inducing points must equal N = n^%d" % X.shape[1])
    n = int(n)

    Z = torch.empty((N,X.shape[1]), device=config.device, dtype=config.dtype)
    grid = torch.meshgrid([torch.linspace(torch.min(X[:,i]), torch.max(X[:,i]), n) for i in range(X.shape[1])], indexing='ij')
    for i in range(X.shape[1]):
        Z[:,i] = grid[i].flatten()
    return Z

def _init_random(N, X):
    sampler = qmc.Halton(d=X.shape[1])
    samples = torch.tensor(sampler.random(n=N), device=config.device, dtype=config.dtype)
    Z = torch.empty((N,X.shape[1]), device=config.device, dtype=config.dtype)
    for i in range(X.shape[1]):
        Z[:,i] = torch.min(X[:,i]) + (torch.max(X[:,i])-torch.min(X[:,i]))*samples[:,i]
    return Z

def _init_density(N, X):
    kernel = gaussian_kde(X.T.detach().cpu().numpy(), bw_method='scott')
    Z = torch.tensor(kernel.resample(N).T, device=config.device, dtype=config.dtype)
    return Z

def init_inducing_points(Z, X, method='grid', output_dims=None):
    """
    Initialize locations for inducing points.

    Args:
        Z (int,list): Number of inducing points. A list of ints of shape (output_dims,) will initialize the specified number of inducing points per output dimension.
        X (torch.tensor): Input data of shape (data_points,input_dims).
        method (str): Method for initialization, can be `grid`, `random`, or `density`.
        output_dims (int): Number of output dimensions. If not None, the first input dimension of the input data must contain the channel IDs.

    Returns:
        torch.tensor: Inducing point locations of shape (data_points,input_dims). In case of multiple output dimensions, the first input dimension will be the channel ID.
    """
    _init = _init_grid
    if method == 'random':
        _init = _init_random
    elif method == 'density':
        _init = _init_density

    if output_dims is not None:
        if isinstance(Z, int) or all(isinstance(z, int) for z in Z) and len(Z) == output_dims:
            if isinstance(Z, int):
                Z = [Z] * output_dims
            M = Z
            Z = torch.zeros((sum(M),X.shape[1]))
            for j in range(len(M)):
                m0 = sum(M[:j])
                m = M[j]
                Z[m0:m0+m,0] = j
                Z[m0:m0+m,1:] = _init(m, X[X[:,0] == j,1:])
    elif isinstance(Z, int):
        M = Z
        Z = _init(M, X)
    return Z

class CholeskyException(Exception):
    def __init__(self, message, K, model):
        self.message = message
        self.K = K
        self.model = model

    def __str__(self):
        return self.message

class Model(torch.nn.Module):
    """
    Base model class.

    Attributes:
        kernel (mogptk.gpr.kernel.Kernel): Kernel.
        likelihood (mogptk.gpr.likelihood.Likelihood): Likelihood.
        mean (mogptk.gpr.mean.Mean): Mean.
    """
    def __init__(self, kernel, X, y, likelihood=GaussianLikelihood(1.0), jitter=1e-8, mean=None):
        super().__init__()

        if not issubclass(type(kernel), Kernel):
            raise ValueError("kernel must derive from mogptk.gpr.Kernel")
        X, y = self._check_input(X, y)
        if mean is not None:
            if not issubclass(type(mean), Mean):
                raise ValueError("mean must derive from mogptk.gpr.Mean")
            mu = mean(X).reshape(-1,1)
            if mu.shape != y.shape:
                raise ValueError("mean and y data must match shapes: %s != %s" % (mu.shape, y.shape))

        if issubclass(type(likelihood), MultiOutputLikelihood) and likelihood.output_dims != kernel.output_dims:
            raise ValueError("kernel and likelihood must have matching output dimensions")
        likelihood.validate_y(X, y)

        # limit to number of significant digits
        if config.dtype == torch.float32:
            jitter = max(jitter, 1e-6)
        elif config.dtype == torch.float64:
            jitter = max(jitter, 1e-15)

        self.kernel = kernel
        self.X = X
        self.y = y
        self.mean = mean
        self.likelihood = likelihood
        self.jitter = jitter
        self.input_dims = X.shape[1]
        self._compiled_forward = None

    def name(self):
        return self.__class__.__name__

    def forward(self, x=None):
        return -self.log_marginal_likelihood() - self.log_prior()

    def compile(self):
        if self._compiled_forward is None:
            self._compiled_forward = torch.jit.trace(self.forward, ())

    def __getstate__(self):
        state = super().__getstate__()
        state['_modules'] = state['_modules'].copy()
        state['_modules'].pop('_compiled_forward', None)
        return state

    def __setattr__(self, name, val):
        if hasattr(self, name) and isinstance(getattr(self, name), Parameter):
            raise AttributeError("parameter is read-only, use Parameter.assign()")
        if isinstance(val, Parameter) and val._name is None:
            val._name = '%s.%s' % (self.__class__.__name__, name)
        elif isinstance(val, torch.nn.ModuleList):
            for i, item in enumerate(val):
                for p in item.parameters():
                    p._name = '%s[%d].%s' % (self.__class__.__name__, i, p._name)
        super().__setattr__(name, val)        

    def _check_input(self, X, y=None):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, device=config.device, dtype=config.dtype)
        else:
            X = X.to(config.device, config.dtype)
        if X.ndim == 0:
            X = X.reshape(1,1)
        elif X.ndim == 1:
            X = X.reshape(-1,1)
        elif X.ndim != 2:
            raise ValueError("X must have dimensions (data_points,input_dims) with input_dims optional")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must not be empty")

        if y is not None:
            if not isinstance(y, torch.Tensor):
                y = torch.tensor(y, device=config.device, dtype=config.dtype)
            else:
                y = y.to(config.device, config.dtype)
            if y.ndim == 0:
                y = y.reshape(1,1)
            elif y.ndim == 1:
                y = y.reshape(-1,1)
            elif y.ndim != 2 or y.shape[1] != 1:
                raise ValueError("y must have one dimension (data_points,)")
            if X.shape[0] != y.shape[0]:
                raise ValueError("number of data points for X and y must match")
            return X, y
        else:
            # X is for prediction
            if X.shape[1] != self.input_dims:
                raise ValueError("X must have %s input dimensions" % self.input_dims)
            return X

    def _index_channel(self, value, X):
        if self.kernel.output_dims is not None and 0 < value.ndim and value.shape[0] == self.kernel.output_dims:
            return torch.index_select(value, dim=0, index=X[:,0].long())
        return value
    
    def print_parameters(self, file=None):
        """
        Print parameters and their values.
        """
        def param_range(lower, upper, train=True, pegged=False):
            if lower is not None:
                if lower.size == 1:
                    lower = lower.item()
                elif (lower.max()-lower.min())/lower.mean() < 1e-6:
                    lower = lower.mean().item()
                else:
                    lower = lower.tolist()
            if upper is not None:
                if upper.size == 1:
                    upper = upper.item()
                elif (upper.max()-upper.min())/upper.mean() < 1e-6:
                    upper = upper.mean().item()
                else:
                    upper = upper.tolist()

            if pegged:
                return "pegged"
            elif not train:
                return "fixed"
            if lower is None and upper is None:
                return "(-∞, ∞)"
            elif lower is None:
                return "(-∞, %s]" % upper
            elif upper is None:
                return "[%s, ∞)" % lower
            return "[%s, %s]" % (lower, upper)

        if file is None:
            try:
                get_ipython  # fails if we're not in a notebook
                table = '<table><tr><th style="text-align:left">Name</th><th>Range</th><th>Value</th></tr>'
                for p in self.parameters():
                    table += '<tr><td style="text-align:left">%s</td><td>%s</td><td>%s</td></tr>' % (p._name, param_range(p.lower, p.upper, p.train, p.pegged), p.numpy())
                table += '</table>'
                display(HTML(table))
                return
            except Exception as e:
                pass

        vals = [["Name", "Range", "Value"]]
        for p in self.parameters():
            vals.append([p._name, param_range(p.lower, p.upper, p.train, p.pegged), p.numpy().tolist()])

        nameWidth = max([len(val[0]) for val in vals])
        rangeWidth = max([len(val[1]) for val in vals])
        for val in vals:
            #print("%-*s  %-*s  %s" % (nameWidth, val[0], rangeWidth, val[1], val[2]), file=file)
            print("%-*s  %s" % (nameWidth, val[0], val[2]), file=file)

    def _cholesky(self, K, add_jitter=False):
        if add_jitter:
            K = K + (self.jitter * K.diagonal().mean()).repeat(K.shape[0]).diagflat()
        try:
            return torch.linalg.cholesky(K)
        except RuntimeError as e:
            print("ERROR:", e.args[0], file=sys.__stdout__)
            if K.isnan().any():
                print("ERROR: kernel matrix has NaNs!", file=sys.__stdout__)
            if K.isinf().any():
                print("ERROR: kernel matrix has infinities!", file=sys.__stdout__)
            self.print_parameters()
            plot_gram(K)
            raise CholeskyException(e.args[0], K, self)

    def log_marginal_likelihood(self):
        """
        Return the log marginal likelihood given by

        $$ \\log p(y) $$

        Returns:
            torch.tensor: Log marginal likelihood.
        """
        raise NotImplementedError()

    def log_prior(self):
        """
        Return the log prior given by

        $$ \\log p(\\theta) $$

        Returns:
            torch.tensor: Log prior.
        """
        return sum([p.log_prior() for p in self.parameters()])

    def loss(self):
        """
        Model loss for training.

        Returns:
            torch.tensor: Loss.
        """
        self.zero_grad(set_to_none=True)
        if self._compiled_forward is None:
            loss = self.forward()
        else:
            loss = self._compiled_forward()
        loss.backward()
        return loss

    def K(self, X1, X2=None):
        """
        Evaluate kernel at `X1` and `X2` and return the NumPy representation.

        Args:
            X1 (torch.tensor): Input of shape (data_points1,input_dims).
            X2 (torch.tensor): Input of shape (data_points2,input_dims).

        Returns:
            numpy.ndarray: Kernel matrix of shape (data_points1,data_points2).
        """
        with torch.inference_mode():
            return self.kernel(X1, X2)

    def predict_f(self, X, full=False):
        """
        Get the predictive posterior for values of f.

        Args:
            X (torch.tensor): Input of shape (data_points,input_dims) needed for multi-output likelihood.
            full (bool): Return the full variance matrix as opposed to the diagonal.

        Returns:
            torch.tensor: Mean of the predictive posterior of shape (data_points,1).
            torch.tensor: Variance of the predictive posterior of shape (data_points,1) or (data_points,data_points).
        """
        raise NotImplementedError()

    def predict_y(self, X, ci=None, sigma=None, n=10000):
        """
        Get the predictive posterior for values of y.

        Args:
            X (torch.tensor): Input of shape (data_points,input_dims) needed for multi-output likelihood.
            ci (list of float): Two percentages [lower, upper] in the range of [0,1] that represent the confidence interval.
            sigma (float): Number of standard deviations of the confidence interval. Only used for short-path for Gaussian likelihood.
            n (int): Number of samples used from distribution to estimate quantile.

        Returns:
            torch.tensor: Mean of the predictive posterior of shape (data_points,1).
            torch.tensor: Lower confidence boundary of the predictive posterior of shape (data_points,1).
            torch.tensor: Upper confidence boundary of the predictive posterior of shape (data_points,1).
        """
        with torch.inference_mode():
            X = self._check_input(X)  # MxD

            mu, var = self.predict_f(X)
            if ci is None and sigma is not None:
                p = 0.5*(1.0 + float(torch.erf(torch.tensor(sigma/np.sqrt(2.0)))))
                ci = [1.0-p, p]
            return self.likelihood.predict(X, mu, var, ci, sigma=sigma, n=n)

    def sample_f(self, Z, n=None, prior=False):
        """
        Sample f values from model.

        Args:
            Z (torch.tensor): Input of shape (data_points,input_dims).
            n (int): Number of samples.
            prior (boolean): Sample from prior instead of posterior.

        Returns:
            torch.tensor: Samples of shape (data_points,n) or (data_points,) if `n` is not given.
        """
        with torch.inference_mode():
            Z = self._check_input(Z)  # MxD

            S = n
            if n is None:
                S = 1

            if prior:
                mu, var = self.mean(Z), self.kernel(Z)
            else:
                mu, var = self.predict_f(Z, full=True)  # Mx1, MxM

            eye = torch.eye(var.shape[0], device=config.device, dtype=config.dtype)
            var += self.jitter * var.diagonal().mean() * eye  # MxM
            samples_f = torch.distributions.multivariate_normal.MultivariateNormal(mu.reshape(-1), var).sample([S])

            if n is None:
                samples_f = samples_f.squeeze()
            return samples_f

    def sample_y(self, Z, n=None):
        """
        Sample y values from model.

        Args:
            Z (torch.tensor): Input of shape (data_points,input_dims).
            n (int): Number of samples.

        Returns:
            torch.tensor: Samples of shape (data_points,n) or (data_points,) if `n` is not given.
        """
        with torch.inference_mode():
            Z = self._check_input(Z)  # MxD

            S = n
            if n is None:
                S = 1

            samples_f = self.sample_f(Z, n=S)
            samples_y = self.likelihood.conditional_sample(Z, samples_f)

            if n is None:
                samples_y = samples_y.squeeze()
            return samples_y

class Exact(Model):
    """
    Regular Gaussian process regression with a Gaussian likelihood which allows for exact inference.

    $$ y \\sim \\mathcal{N}(0, K + \\sigma^2I) $$

    Args:
        kernel (mpgptk.gpr.kernel.Kernel): Kernel.
        X (torch.tensor): Input data of shape (data_points,input_dims).
        y (torch.tensor): Output data of shape (data_points,).
        variance (float,torch.tensor): Gaussian likelihood initial variance. Passing a float will train a single variance for all channels. Passing a tensor of shape (channels,) will assign and train different variances per multi-output channel.
        data_variance (torch.tensor): Assign different and fixed variances per data point of shape (data_points,). These are added to the kernel's diagonal while still training an additional Gaussian variance.
        jitter (float): Relative jitter of the diagonal's mean added to the kernel's diagonal before calculating the Cholesky.
        mean (mogptk.gpr.mean.Mean): Mean.
    """
    def __init__(self, kernel, X, y, variance=1.0, data_variance=None, jitter=1e-8, mean=None):
        if data_variance is not None:
            data_variance = Parameter.to_tensor(data_variance)
            if data_variance.ndim != 1 or X.ndim == 2 and data_variance.shape[0] != X.shape[0]:
                raise ValueError("data variance must have shape (data_points,)")
            data_variance = data_variance.diagflat()
        self.data_variance = data_variance

        variance = Parameter.to_tensor(variance)
        channels = 1
        if kernel.output_dims is not None:
            channels = kernel.output_dims
        if 1 < variance.ndim or variance.ndim == 1 and variance.shape[0] != channels:
            raise ValueError("variance must be float or have shape (channels,)")

        super().__init__(kernel, X, y, GaussianLikelihood(torch.sqrt(variance)), jitter, mean)

        self.eye = torch.eye(self.X.shape[0], device=config.device, dtype=config.dtype)
        self.log_marginal_likelihood_constant = 0.5*self.X.shape[0]*np.log(2.0*np.pi)

    def log_marginal_likelihood(self):
        Kff = self.kernel.K(self.X)
        Kff += self._index_channel(self.likelihood.scale().square(), self.X) * self.eye  # NxN
        if self.data_variance is not None:
            Kff += self.data_variance
        L = self._cholesky(Kff, add_jitter=True)  # NxN

        if self.mean is not None:
            y = self.y - self.mean(self.X).reshape(-1,1)  # Nx1
        else:
            y = self.y  # Nx1

        p = -self.log_marginal_likelihood_constant
        p -= L.diagonal().log().sum() # 0.5 is taken inside the log: L is the square root
        p -= 0.5*y.T.mm(torch.cholesky_solve(y,L)).squeeze()
        return p

    def predict_f(self, X, full=False):
        with torch.inference_mode():
            X = self._check_input(X)  # MxD
            if self.mean is not None:
                y = self.y - self.mean(self.X).reshape(-1,1)  # Nx1
            else:
                y = self.y  # Nx1

            Kff = self.kernel.K(self.X)
            Kff += self._index_channel(self.likelihood.scale().square(), self.X) * self.eye  # NxN
            if self.data_variance is not None:
                Kff += self.data_variance
            Kfs = self.kernel.K(self.X,X)  # NxM

            Lff = self._cholesky(Kff, add_jitter=True)  # NxN
            v = torch.linalg.solve_triangular(Lff,Kfs,upper=False)  # NxM

            mu = Kfs.T.mm(torch.cholesky_solve(y,Lff))  # Mx1
            if self.mean is not None:
                mu += self.mean(X).reshape(-1,1)  # Mx1

            if full:
                Kss = self.kernel.K(X)  # MxM
                var = Kss - v.T.mm(v)  # MxM
            else:
                Kss_diag = self.kernel.K_diag(X)  # M
                var = Kss_diag - v.T.square().sum(dim=1)  # M
                var = var.reshape(-1,1)
            return mu, var

class Snelson(Model):
    """
    A sparse Gaussian process regression based on Snelson and Ghahramani [1] with a Gaussian likelihood and inducing points.

    Args:
        kernel (mogptk.gpr.kernel.Kernel): Kernel.
        X (torch.tensor): Input data of shape (data_points,input_dims).
        y (torch.tensor): Output data of shape (data_points,).
        Z (int,torch.tensor): Number of inducing points to be distributed over the input space. Passing a tensor of shape (inducing_points,input_dims) sets the initial positions of the inducing points.
        Z_init (str): Method for initialization of inducing points, can be `grid`, `random`, or `density`.
        variance (float,torch.tensor): Gaussian likelihood initial variance. Passing a float will train a single variance for all channels. Passing a tensor of shape (channels,) will assign and train different variances per multi-output channel.
        jitter (float): Relative jitter of the diagonal's mean added to the kernel's diagonal before calculating the Cholesky.
        mean (mogptk.gpr.mean.Mean): Mean.

    [1] E. Snelson, Z. Ghahramani, "Sparse Gaussian Processes using Pseudo-inputs", 2005
    """
    def __init__(self, kernel, X, y, Z=10, Z_init='grid', variance=1.0, jitter=1e-8, mean=None):
        variance = Parameter.to_tensor(variance).squeeze()
        if 1 < variance.ndim or variance.ndim == 1 and variance.shape[0] != kernel.output_dims:
            raise ValueError("variance must be float or have shape (channels,)")

        super().__init__(kernel, X, y, GaussianLikelihood(torch.sqrt(variance)), jitter, mean)

        Z = init_inducing_points(Z, self.X, method=Z_init, output_dims=kernel.output_dims)
        Z = self._check_input(Z)
        
        self.eye = torch.eye(Z.shape[0], device=config.device, dtype=config.dtype)
        self.log_marginal_likelihood_constant = 0.5*self.X.shape[0]*np.log(2.0*np.pi)
        self.Z = Parameter(Z, name="induction_points")
        if kernel.output_dims is not None:
            self.Z.num_parameters -= self.Z().shape[0]

    def log_marginal_likelihood(self):
        if self.mean is not None:
            y = self.y - self.mean(self.X).reshape(-1,1)  # Nx1
        else:
            y = self.y  # Nx1

        Kff_diag = self.kernel.K_diag(self.X)  # N
        Kuf = self.kernel.K(self.Z(),self.X)  # MxN
        Kuu = self.kernel.K(self.Z())  # MxM

        Luu = self._cholesky(Kuu, add_jitter=True)  # MxM;  Luu = Kuu^(1/2)
        v = torch.linalg.solve_triangular(Luu,Kuf,upper=False)  # MxN;  Kuu^(-1/2).Kuf
        g = Kff_diag - v.T.square().sum(dim=1) + self._index_channel(self.likelihood.scale().square(), self.X)  # N;  diag(Kff-Qff) + sigma^2.I
        G = torch.diagflat(1.0/g)  # N
        L = self._cholesky(v.mm(G).mm(v.T) + self.eye)  # MxM;  (Kuu^(-1/2).Kuf.G.Kfu.Kuu^(-1/2) + I)^(1/2)

        c = torch.linalg.solve_triangular(L,v.mm(G).mm(y),upper=False)  # Mx1;  L^(-1).Kuu^(-1/2).Kuf.G.y

        p = -self.log_marginal_likelihood_constant
        p -= L.diagonal().log().sum() # 0.5 is taken as the square root of L
        p -= 0.5*g.log().sum()
        p -= 0.5*y.T.mm(G).mm(y).squeeze()
        p += 0.5*c.T.mm(c).squeeze()
        return p

    def predict_f(self, X, full=False):
        with torch.inference_mode():
            X = self._check_input(X)  # MxD
            if self.mean is not None:
                y = self.y - self.mean(self.X).reshape(-1,1)  # Nx1
            else:
                y = self.y  # Nx1

            Kff_diag = self.kernel.K_diag(self.X)  # N
            Kuf = self.kernel.K(self.Z(),self.X)  # MxN
            Kuu = self.kernel.K(self.Z())  # MxM
            Kus = self.kernel.K(self.Z(),X)  # MxS

            Luu = self._cholesky(Kuu, add_jitter=True)  # MxM;  Kuu^(1/2)
            v = torch.linalg.solve_triangular(Luu,Kuf,upper=False)  # MxN;  Kuu^(-1/2).Kuf
            g = Kff_diag - v.T.square().sum(dim=1) + self._index_channel(self.likelihood.scale().square(), self.X)
            G = torch.diagflat(1.0/g)  # N
            L = self._cholesky(v.mm(G).mm(v.T) + self.eye)  # MxM;  (Kuu^(-1/2).Kuf.G.Kfu.Kuu^(-1/2) + I)^(1/2)

            a = torch.linalg.solve_triangular(Luu,Kus,upper=False)  # NxM
            b = torch.linalg.solve_triangular(L,a,upper=False)
            c = torch.linalg.solve_triangular(L,v.mm(G).mm(y),upper=False)  # Mx1;  L^(-1).Kuu^(-1/2).Kuf.G.y

            mu = b.T.mm(c)  # Mx1
            if self.mean is not None:
                mu += self.mean(X).reshape(-1,1)  # Mx1

            if full:
                Kss = self.kernel(X)  # MxM
                var = Kss - a.T.mm(w) + b.T.mm(u)  # MxM
            else:
                Kss_diag = self.kernel.K_diag(X)  # M
                var = Kss_diag - a.T.square().sum(dim=1) + b.T.square().sum(dim=1)  # M
                var = var.reshape(-1,1)
            return mu, var

class OpperArchambeau(Model):
    """
    A Gaussian process regression based on Opper and Archambeau [1] with a non-Gaussian likelihood.

    Args:
        kernel (mogptk.gpr.kernel.Kernel): Kernel.
        X (torch.tensor): Input data of shape (data_points,input_dims).
        y (torch.tensor): Output data of shape (data_points,).
        likelihood (mogptk.gpr.likelihood.Likelihood): Likelihood.
        jitter (float): Relative jitter of the diagonal's mean added to the kernel's diagonal before calculating the Cholesky.
        mean (mogptk.gpr.mean.Mean): Mean.

    [1] M. Opper, C. Archambeau, "The Variational Gaussian Approximation Revisited", 2009
    """
    def __init__(self, kernel, X, y, likelihood=GaussianLikelihood(1.0),
                 jitter=1e-8, mean=None):
        super().__init__(kernel, X, y, likelihood, jitter, mean)

        n = self.X.shape[0]
        self.eye = torch.eye(n, device=config.device, dtype=config.dtype)
        self.q_nu = Parameter(torch.zeros(n,1))
        self.q_lambda = Parameter(torch.ones(n,1), lower=config.positive_minimum)
        self.likelihood = likelihood

    def elbo(self):
        if self.mean is not None:
            y = self.y - self.mean(self.X).reshape(-1,1)  # Nx1
        else:
            y = self.y  # Nx1

        q_nu = self.q_nu()
        q_lambda = self.q_lambda()

        Kff = self.kernel(self.X)  # NxN
        L = self._cholesky(q_lambda*q_lambda.T*Kff + self.eye)
        invL = torch.linalg.solve_triangular(L,self.eye,upper=False)  # NxN

        qf_mu = Kff.mm(q_nu)
        qf_var_diag = 1.0/q_lambda.square() - (invL.T.mm(invL)/q_lambda/q_lambda.T).diagonal().reshape(-1,1)

        kl = q_nu.T.mm(qf_mu).squeeze()  # Mahalanobis
        kl += L.diagonal().square().log().sum()  # determinant TODO: is this correct?
        #kl += invL.diagonal().square().sum()  # trace
        kl += invL.square().sum()  # trace
        kl -= q_nu.shape[0]

        if self.mean is not None:
            qf_mu = qf_mu - self.mean(self.X).reshape(-1,1)  # Sx1
        var_exp = self.likelihood.variational_expectation(self.X, y, qf_mu, qf_var_diag)

        #eye = torch.eye(q_lambda.shape[0], device=config.device, dtype=config.dtype)
        #qf_var = (1.0/q_lambda.square())*eye - invL.T.mm(invL)/q_lambda/q_lambda.T
        #kl = -q_nu.shape[0]
        #kl += q_nu.T.mm(qf_mu).squeeze()  # Mahalanobis
        #kl -= qf_var.det().log()  # determinant
        #kl += invL.diagonal().square().sum()  # trace

        #kl = -q_nu.shape[0]
        #kl += q_nu.T.mm(q_nu).squeeze()  # Mahalanobis
        #kl -= qf_var.det().log()  # determinant
        #kl += qf_var_diag.sum()  # trace
        return var_exp - 0.5*kl

    def log_marginal_likelihood(self):
        # maximize the lower bound
        return self.elbo()

    def predict_f(self, X, full=False):
        with torch.inference_mode():
            X = self._check_input(X)  # MxD

            Kff = self.kernel(self.X)
            Kfs = self.kernel(self.X,X)  # NxS

            L = self._cholesky(Kff + (1.0/self.q_lambda().square()).diagflat())  # NxN
            a = torch.linalg.solve_triangular(L,Kfs,upper=False)  # NxS;  Kuu^(-1/2).Kus

            mu = Kfs.T.mm(self.q_nu())  # Sx1
            if self.mean is not None:
                mu += self.mean(X).reshape(-1,1)  # Sx1

            if full:
                Kss = self.kernel(X)  # SxS
                var = Kss - a.T.mm(a)  # SxS
            else:
                Kss_diag = self.kernel.K_diag(X)  # M
                var = Kss_diag - a.T.square().sum(dim=1)  # M
                var = var.reshape(-1,1)
            return mu, var

class Titsias(Model):
    """
    A sparse Gaussian process regression based on Titsias [1] with a Gaussian likelihood.

    Args:
        kernel (mogptk.gpr.kernel.Kernel): Kernel.
        X (torch.tensor): Input data of shape (data_points,input_dims).
        y (torch.tensor): Output data of shape (data_points,).
        Z (int,torch.tensor): Number of inducing points to be distributed over the input space. Passing a tensor of shape (inducing_points,input_dims) sets the initial positions of the inducing points.
        Z_init (str): Method for initialization of inducing points, can be `grid`, `random`, or `density`.
        variance (float): Gaussian likelihood initial variance.
        jitter (float): Relative jitter of the diagonal's mean added to the kernel's diagonal before calculating the Cholesky.
        mean (mogptk.gpr.mean.Mean): Mean.

    [1] Titsias, "Variational learning of induced variables in sparse Gaussian processes", 2009
    """
    # See: http://krasserm.github.io/2020/12/12/gaussian-processes-sparse/
    def __init__(self, kernel, X, y, Z, Z_init='grid', variance=1.0, jitter=1e-8, mean=None):
        # TODO: variance per channel
        variance = Parameter.to_tensor(variance)

        super().__init__(kernel, X, y, GaussianLikelihood(torch.sqrt(variance)), jitter, mean)

        Z = init_inducing_points(Z, self.X, method=Z_init, output_dims=kernel.output_dims)
        Z = self._check_input(Z)

        self.eye = torch.eye(Z.shape[0], device=config.device, dtype=config.dtype)
        self.log_marginal_likelihood_constant = 0.5*self.X.shape[0]*np.log(2.0*np.pi)
        self.Z = Parameter(Z, name="induction_points")
        if kernel.output_dims is not None:
            self.Z.num_parameters -= self.Z().shape[0]

    def elbo(self):
        if self.mean is not None:
            y = self.y - self.mean(self.X).reshape(-1,1)  # Nx1
        else:
            y = self.y  # Nx1

        Kff_diag = self.kernel.K_diag(self.X)  # N
        Kuf = self.kernel(self.Z(),self.X)  # MxN
        Kuu = self.kernel(self.Z())  # MxM

        Luu = self._cholesky(Kuu, add_jitter=True)  # MxM;  Kuu^(1/2)
        v = torch.linalg.solve_triangular(Luu,Kuf,upper=False)  # MxN;  Kuu^(-1/2).Kuf
        Q = v.mm(v.T)  # MxM;  Kuu^(-1/2).Kuf.Kfu.Kuu^(-1/2)
        L = self._cholesky(Q/self.likelihood.scale().square() + self.eye)  # MxM;  (Q/sigma^2 + I)^(1/2)

        c = torch.linalg.solve_triangular(L,v.mm(y),upper=False)/self.likelihood.scale().square()  # Mx1;  L^(-1).Kuu^(-1/2).Kuf.y

        # p = log N(0, Kfu.Kuu^(-1).Kuf + I/sigma^2) - 1/(2.sigma^2).Trace(Kff - Kfu.Kuu^(-1).Kuf)
        p = -self.log_marginal_likelihood_constant
        p -= L.diagonal().log().sum() # 0.5 is taken as the square root of L
        p -= self.X.shape[0]*self.likelihood.scale().log()
        p -= 0.5*y.T.mm(y).squeeze()/self.likelihood.scale().square()
        p += 0.5*c.T.mm(c).squeeze()
        p -= 0.5*(Kff_diag.sum() - Q.trace())/self.likelihood.scale().square() # trace
        return p

    def log_marginal_likelihood(self):
        # maximize the lower bound
        return self.elbo()

    def predict_f(self, X, full=False):
        with torch.inference_mode():
            X = self._check_input(X)  # MxD
            if self.mean is not None:
                y = self.y - self.mean(self.X).reshape(-1,1)  # Nx1
            else:
                y = self.y  # Nx1

            Kus = self.kernel(self.Z(),X)  # MxS
            Kuf = self.kernel(self.Z(),self.X)  # MxN
            Kuu = self.kernel(self.Z())  # MxM

            Luu = self._cholesky(Kuu, add_jitter=True)  # MxM;  Kuu^(1/2)
            v = torch.linalg.solve_triangular(Luu,Kuf,upper=False)  # MxN;  Kuu^(-1/2).Kuf
            L = self._cholesky(v.mm(v.T)/self.likelihood.scale().square() + self.eye)  # MxM;  (Kuu^(-1/2).Kuf.Kfu.Kuu^(-1/2)/sigma^2 + I)^(1/2)

            a = torch.linalg.solve_triangular(Luu,Kus,upper=False)  # MxS;  Kuu^(-1/2).Kus
            b = torch.linalg.solve_triangular(L,a,upper=False)  # MxS;  L^(-1).Kuu^(-1/2).Kus
            c = torch.linalg.solve_triangular(L,v.mm(y),upper=False)/self.likelihood.scale().square()  # Mx1;  L^(-1).Kuu^(-1/2).Kuf.y

            # mu = sigma^(-2).Ksu.Kuu^(-1/2).(sigma^(-2).Kuu^(-1/2).Kuf.Kfu.Kuu^(-1/2) + I)^(-1).Kuu^(-1/2).Kuf.y
            mu = b.T.mm(c)  # Mx1
            if self.mean is not None:
                mu += self.mean(X).reshape(-1,1)  # Mx1

            # var = Kss - Qsf.(Qff + sigma^2 I)^(-1).Qfs
            # below is the equivalent but more stable version by using the matrix inversion lemma
            # var = Kss - Ksu.Kuu^(-1).Kus + Ksu.Kuu^(-1/2).(sigma^(-2).Kuu^(-1/2).Kuf.Kfu.Kuu^(-1/2) + I)^(-1).Kuu^(-1/2).Kus
            if full:
                Kss = self.kernel(X)  # MxM
                var = Kss - a.T.mm(a) + b.T.mm(b)  # MxM
            else:
                Kss_diag = self.kernel.K_diag(X)  # M
                var = Kss_diag - a.T.square().sum(dim=1) + b.T.square().sum(dim=1)  # M
                var = var.reshape(-1,1)
            return mu, var

class SparseHensman(Model):
    """
    A sparse Gaussian process regression based on Hensman et al. [1] with a non-Gaussian likelihood.

    Args:
        kernel (mogptk.gpr.kernel.Kernel): Kernel.
        X (torch.tensor): Input data of shape (data_points,input_dims).
        y (torch.tensor): Output data of shape (data_points,).
        Z (int,torch.tensor): Number of inducing points to be distributed over the input space. Passing a tensor of shape (inducing_points,input_dims) sets the initial positions of the inducing points.
        Z_init (str): Method for initialization of inducing points, can be `grid`, `random`, or `density`.
        likelihood (mogptk.gpr.likelihood.Likelihood): Likelihood.
        jitter (float): Relative jitter of the diagonal's mean added to the kernel's diagonal before calculating the Cholesky.
        mean (mogptk.gpr.mean.Mean): Mean.

    [1] J. Hensman, et al., "Scalable Variational Gaussian Process Classification", 2015
    """
    # This version replaces mu_q by L.mu_q and sigma_q by L.sigma_q.L^T, where LL^T = Kuu
    # So that p(u) ~ N(0,1) and q(u) ~ N(L.mu_q, L.sigma_q.L^T)
    def __init__(self, kernel, X, y, Z=None, Z_init='grid', likelihood=GaussianLikelihood(1.0),
                 jitter=1e-8, mean=None):
        super().__init__(kernel, X, y, likelihood, jitter, mean)

        n = self.X.shape[0]
        self.is_sparse = Z is not None
        if self.is_sparse:
            Z = init_inducing_points(Z, self.X, method=Z_init, output_dims=kernel.output_dims)
            Z = self._check_input(Z)
            n = Z.shape[0]

        self.eye = torch.eye(n, device=config.device, dtype=config.dtype)
        self.log_marginal_likelihood_constant = 0.5*self.X.shape[0]*np.log(2.0*np.pi)
        self.q_mu = Parameter(torch.zeros(n,1))
        self.q_sqrt = Parameter(torch.eye(n))
        self.q_sqrt.num_parameters = int((n*n+n)/2)
        if self.is_sparse:
            self.Z = Parameter(Z, name="induction_points")
            if kernel.output_dims is not None:
                self.Z.num_parameters -= self.Z().shape[0]
        else:
            self.Z = Parameter(self.X, train=False)  # don't use inducing points

    def kl_gaussian(self, q_mu, q_sqrt):
        S_diag = q_sqrt.diagonal().square() # NxN
        kl = q_mu.T.mm(q_mu).squeeze()  # Mahalanobis
        kl -= S_diag.log().sum()  # determinant of q_var
        kl += S_diag.sum()  # same as Trace(p_var^(-1).q_var)
        kl -= q_mu.shape[0]
        return 0.5*kl

    def elbo(self):
        if self.mean is not None:
            y = self.y - self.mean(self.X).reshape(-1,1)  # Nx1
        else:
            y = self.y  # Nx1

        if self.is_sparse:
            qf_mu, qf_var_diag = self._predict_f(self.X, full=False)
        else:
            Kff = self.kernel(self.X)
            Lff = self._cholesky(Kff, add_jitter=True)  # NxN

            qf_mu = Lff.mm(self.q_mu())
            if self.mean is not None:
                qf_mu -= self.mean(self.X).reshape(-1,1)  # Sx1

            qf_sqrt = Lff.mm(self.q_sqrt().tril())
            qf_var_diag = qf_sqrt.mm(qf_sqrt.T).diagonal().reshape(-1,1)

        var_exp = self.likelihood.variational_expectation(self.X, y, qf_mu, qf_var_diag)
        kl = self.kl_gaussian(self.q_mu(), self.q_sqrt())
        return var_exp - kl

    def log_marginal_likelihood(self):
        # maximize the lower bound
        return self.elbo()

    def _predict_f(self, X, full=False):
        Kuu = self.kernel(self.Z())
        Kus = self.kernel(self.Z(),X)  # NxS

        Luu = self._cholesky(Kuu, add_jitter=True)  # NxN
        a = torch.linalg.solve_triangular(Luu,Kus,upper=False)  # NxS;  Kuu^(-1/2).Kus
        b = self.q_sqrt().tril().T.mm(torch.linalg.solve_triangular(Luu,Kus,upper=False))

        mu = Kus.T.mm(torch.linalg.solve_triangular(Luu.T,self.q_mu(),upper=True))  # Sx1
        if full:
            Kss = self.kernel(X)  # SxS
            var = Kss - a.T.mm(a) + b.T.mm(b)  # SxS
        else:
            Kss_diag = self.kernel.K_diag(X)  # M
            var = Kss_diag - a.T.square().sum(dim=1) + b.T.square().sum(dim=1)  # M
            var = var.reshape(-1,1)
        return mu, var

    def predict_f(self, X, full=False):
        with torch.inference_mode():
            X = self._check_input(X)  # MxD

            mu, var = self._predict_f(X, full=full)
            if self.mean is not None:
                mu += self.mean(X).reshape(-1,1)  # Mx1
            return mu, var


class Hensman(SparseHensman):
    """
    A Gaussian process regression based on Hensman et al. [1] with a non-Gaussian likelihood. This is equivalent to the `SparseHensman` model if we set the inducing points equal to the data points and by turning off training the inducing points.

    Args:
        kernel (mogptk.gpr.kernel.Kernel): Kernel.
        X (torch.tensor): Input data of shape (data_points,input_dims).
        y (torch.tensor): Output data of shape (data_points,).
        likelihood (mogptk.gpr.likelihood.Likelihood): Likelihood.
        jitter (float): Relative jitter of the diagonal's mean added to the kernel's diagonal before calculating the Cholesky.
        mean (mogptk.gpr.mean.Mean): Mean.

    [1] J. Hensman, et al., "Scalable Variational Gaussian Process Classification", 2015
    """
    def __init__(self, kernel, X, y, likelihood=GaussianLikelihood(1.0), jitter=1e-8, mean=None):
        super().__init__(kernel, X, y, None, 'grid', likelihood, jitter, mean)
