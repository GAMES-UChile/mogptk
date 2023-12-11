import torch
import numpy as np
from . import config, Parameter

def identity(x):
    """
    The identity link function given by

    $$ y = x $$
    """
    return x

def square(x):
    """
    The square link function given by

    $$ y = x^2 $$
    """
    return torch.square(x)

def exp(x):
    """
    The exponential link function given by

    $$ y = e^{x} $$
    """
    return torch.exp(x)

def probit(x):
    """
    The probit link function given by

    $$ y = \\sqrt{2} \\operatorname{erf}^{-1}(2x-1)\\right) $$
    """
    return np.sqrt(2)*torch.erfinv(2.0*x - 1.0)

def inv_probit(x):
    """
    The inverse probit link function given by

    $$ y = \\frac{1}{2} \\left(1 + \\operatorname{erf}(x/\\sqrt{2})\\right) $$
    """
    jitter = 1e-3
    return 0.5*(1.0+torch.erf(x/np.sqrt(2.0))) * (1.0-2.0*jitter) + jitter

# also inv_logit or logistic
def sigmoid(x):
    """
    The logistic, inverse logit, or sigmoid link function given by

    $$ y = \\frac{1}{1 + e^{-x}} $$
    """
    return 1.0/(1.0+torch.exp(-x))

def log_logistic_distribution(loc, scale):
    return torch.distributions.transformed_distribution.TransformedDistribution(
        base_distribution=torch.distributions.uniform.Uniform(0.0,1.0),
        transforms=[
            torch.distributions.transforms.SigmoidTransform().inv,
            torch.distributions.transforms.AffineTransform(loc=loc, scale=scale),
            torch.distributions.transforms.ExpTransform(),
        ],
    )

class GaussHermiteQuadrature:
    def __init__(self, deg=20, t_scale=None, w_scale=None):
        t, w = np.polynomial.hermite.hermgauss(deg)
        t = t.reshape(-1,1)
        w = w.reshape(-1,1)
        if t_scale is not None:
            t *= t_scale
        if w_scale is not None:
            w *= w_scale
        self.t = torch.tensor(t, device=config.device, dtype=config.dtype)  # degx1
        self.w = torch.tensor(w, device=config.device, dtype=config.dtype)  # degx1
        self.deg = deg

    def __call__(self, mu, var, F):
        return F(mu + var.sqrt().mm(self.t.T)).mm(self.w)  # Nx1

class Likelihood(torch.nn.Module):
    """
    Base likelihood.

    Args:
        quadratures (int): Number of quadrature points to use when approximating using Gauss-Hermite quadratures.
    """
    def __init__(self, quadratures=20):
        super().__init__()
        self.quadrature = GaussHermiteQuadrature(deg=quadratures, t_scale=np.sqrt(2), w_scale=1.0/np.sqrt(np.pi))
        self.output_dims = None

    def name(self):
        return self.__class__.__name__

    def __setattr__(self, name, val):
        if name == 'train':
            for p in self.parameters():
                p.train = val
            return
        if hasattr(self, name) and isinstance(getattr(self, name), Parameter):
            raise AttributeError("parameter is read-only, use Parameter.assign()")
        if isinstance(val, Parameter) and val._name is None:
            val._name = '%s.%s' % (self.__class__.__name__, name)
        elif isinstance(val, torch.nn.ModuleList):
            for i, item in enumerate(val):
                for p in item.parameters():
                    p._name = '%s[%d].%s' % (self.__class__.__name__, i, p._name)
        super().__setattr__(name, val)

    def _channel_indices(self, X):
        c = X[:,0].long()
        m = [c==j for j in range(self.output_dims)]
        r = [torch.nonzero(m[j], as_tuple=False).reshape(-1) for j in range(self.output_dims)]
        return r

    def validate_y(self, X, y):
        """
        Validate whether the y input is within the likelihood's support.
        """
        pass

    def log_prob(self, X, y, f):
        """
        Calculate the log probability density

        $$ \\log(p(y|f)) $$

        with \\(p(y|f)\\) our likelihood.

        Args:
            X (torch.tensor): Input of shape (data_points,input_dims).
            y (torch.tensor): Values for y of shape (data_points,).
            f (torch.tensor): values for f of shape (data_points,quadratures).

        Returns:
            torch.tensor: Log probability density of shape (data_points,quadratures).
        """
        raise NotImplementedError()

    def variational_expectation(self, X, y, mu, var):
        """
        Calculate the variational expectation

        $$ \\int \\log(p(y|X,f)) \\; q(f) \\; df $$

        where \\(q(f) \\sim \\mathcal{N}(\\mu,\\Sigma)\\) and \\(p(y|f)\\) our likelihood. By default this uses Gauss-Hermite quadratures to approximate the integral.

        Args:
            X (torch.tensor): Input of shape (data_points,input_dims).
            y (torch.tensor): Values for y of shape (data_points,).
            mu (torch.tensor): Mean of the posterior \\(q(f)\\) of shape (data_points,1).
            var (torch.tensor): Variance of the posterior \\(q(f)\\) of shape (data_points,1).

        Returns:
            torch.tensor: Expected log density of shape ().
        """
        q = self.quadrature(mu, var, lambda f: self.log_prob(X,y,f))
        return q.sum()

    def conditional_mean(self, X, f):
        """
        Calculate the mean of the likelihood conditional on \\(f\\).

        Args:
            X (torch.tensor): Input of shape (data_points,input_dims).
            f (torch.tensor): Posterior values for f of shape (data_points,quadratures).

        Returns:
            torch.tensor: Mean of the predictive posterior \\(p(y|f)\\) of shape (data_points,quadratures).
        """
        raise NotImplementedError()

    def conditional_sample(self, X, f):
        """
        Sample from likelihood distribution.

        Args:
            X (torch.tensor): Input of shape (data_points,input_dims).
            f (torch.tensor): Samples of f values of shape (n,data_points).

        Returns:
            torch.tensor: Samples of shape (n,data_points).
        """
        # f: nxN
        raise NotImplementedError()

    def predict(self, X, mu, var, ci=None, sigma=None, n=10000):
        """
        Calculate the mean and variance of the predictive distribution

        $$ \\mu = \\iint y \\; p(y|f) \\; q(f) \\; df dy $$
        $$ \\Sigma = \\iint y^2 \\; p(y|f) \\; q(f) \\; df dy - \\mu^2 $$

        where \\(q(f) \\sim \\mathcal{N}(\\mu,\\Sigma)\\) and \\(p(y|f)\\) our likelihood. By default this uses Gauss-Hermite quadratures to approximate both integrals.

        Args:
            X (torch.tensor): Input of shape (data_points,input_dims).
            mu (torch.tensor): Mean of the posterior \\(q(f)\\) of shape (data_points,1).
            var (torch.tensor): Variance of the posterior \\(q(f)\\) of shape (data_points,1).
            ci (list of float): Two percentages [lower, upper] in the range of [0,1] that represent the confidence interval.
            sigma (float): Number of standard deviations of the confidence interval. Only used for short-path for Gaussian likelihood.
            n (int): Number of samples used from distribution to estimate quantile.

        Returns:
            torch.tensor: Mean of the predictive posterior \\(p(y|f)\\) of shape (data_points,1).
            torch.tensor: Lower confidence boundary of shape (data_points,1).
            torch.tensor: Upper confidence boundary of shape (data_points,1).
        """
        # mu,var: Nx1
        mu = self.quadrature(mu, var, lambda f: self.conditional_mean(X,f))
        if ci is None:
            return mu

        samples_f = torch.distributions.normal.Normal(mu, var).sample([n]) # nxNx1
        samples_y = self.conditional_sample(X, samples_f) # nxNx1
        if samples_y is None:
            return mu, mu, mu
        samples_y, _ = samples_y.sort(dim=0)
        lower = int(ci[0]*n + 0.5)
        upper = int(ci[1]*n + 0.5)
        return mu, samples_y[lower,:], samples_y[upper,:]

class MultiOutputLikelihood(Likelihood):
    """
    Multi-output likelihood to assign a different likelihood per channel.

    Args:
        likelihoods (mogptk.gpr.likelihood.Likelihood): List of likelihoods equal to the number of output dimensions.
    """
    def __init__(self, *likelihoods):
        super().__init__()

        if isinstance(likelihoods, tuple):
            if len(likelihoods) == 1 and isinstance(likelihoods[0], list):
                likelihoods = likelihoods[0]
            else:
                likelihoods = list(likelihoods)
        elif not isinstance(likelihoods, list):
            likelihoods = [likelihoods]
        if len(likelihoods) == 0:
            raise ValueError("must pass at least one likelihood")
        for i, likelihood in enumerate(likelihoods):
            if not issubclass(type(likelihood), Likelihood):
                raise ValueError("must pass likelihoods")
            elif isinstance(likelihood, MultiOutputLikelihood):
                raise ValueError("can not nest MultiOutputLikelihoods")

        self.output_dims = len(likelihoods)
        self.likelihoods = torch.nn.ModuleList(likelihoods)

    def name(self):
        names = [likelihood.name() for likelihood in self.likelihoods]
        return '[%s]' % (','.join(names),)

    def validate_y(self, X, y):
        if self.output_dims == 1:
            self.likelihoods[0].validate_y(X, y)
            return

        r = self._channel_indices(X)
        for i in range(self.output_dims):
            self.likelihoods[i].validate_y(X, y[r[i],:])

    def log_prob(self, X, y, f):
        # y: Nx1
        # f: NxQ
        r = self._channel_indices(X)
        res = torch.empty(f.shape, device=config.device, dtype=config.dtype)
        for i in range(self.output_dims):
            res[r[i],:] = self.likelihoods[i].log_prob(X, y[r[i],:], f[r[i],:])
        return res  # NxQ

    def variational_expectation(self, X, y, mu, var):
        # y,mu,var: Nx1
        q = torch.tensor(0.0, dtype=config.dtype, device=config.device)
        r = self._channel_indices(X)
        for i in range(self.output_dims):
            q += self.likelihoods[i].variational_expectation(X, y[r[i],:], mu[r[i],:], var[r[i],:]).sum()  # sum over N
        return q

    def conditional_mean(self, X, f):
        # f: NxQ
        r = self._channel_indices(X)
        res = torch.empty(f.shape, device=config.device, dtype=config.dtype)
        for i in range(self.output_dims):
            res[r[i],:] = self.likelihoods[i].conditional_mean(X, f[r[i],:])
        return res  # NxQ

    def conditional_sample(self, X, f):
        # f: nxN
        r = self._channel_indices(X)
        for i in range(self.output_dims):
            f[:,r[i]] = self.likelihoods[i].conditional_sample(X, f[:,r[i]])
        return f  # nxN

    def predict(self, X, mu, var, ci=None, sigma=None, n=10000):
        # mu,var: Nx1
        r = self._channel_indices(X)
        res = torch.empty(mu.shape, device=config.device, dtype=config.dtype)
        if ci is None:
            for i in range(self.output_dims):
                res[r[i],:] = self.likelihoods[i].predict(X, mu[r[i],:], var[r[i],:], ci=ci, sigma=sigma, n=n)
            return res

        lower = torch.empty(mu.shape, device=config.device, dtype=config.dtype)
        upper = torch.empty(mu.shape, device=config.device, dtype=config.dtype)
        for i in range(self.output_dims):
            res[r[i],:], lower[r[i],:], upper[r[i],:] = self.likelihoods[i].predict(X, mu[r[i],:], var[r[i],:], ci=ci, sigma=sigma, n=n)
        return res, lower, upper  # Nx1

class GaussianLikelihood(Likelihood):
    """
    Gaussian likelihood given by

    $$ p(y|f) = \\frac{1}{\\sigma\\sqrt{2\\pi}}e^{-\\frac{(y-f)^2}{2\\sigma^2}} $$

    with \\(\\sigma\\) the scale.

    Args:
        scale (float,torch.tensor): Scale as float or as tensor of shape (output_dims,) when considering a multi-output Gaussian likelihood.

    Attributes:
        scale (mogptk.gpr.parameter.Parameter): Scale \\(\\sigma\\).
    """
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = Parameter(scale, lower=config.positive_minimum)
        if self.scale.ndim == 1:
            self.output_dims = self.scale.shape[0]

    def log_prob(self, X, y, f):
        # y: Nx1
        # f: NxQ
        p = -0.5 * (np.log(2.0*np.pi) + 2.0*self.scale().log() + ((y-f)/self.scale()).square())
        return p  # NxQ

    def variational_expectation(self, X, y, mu, var):
        # y,mu,var: Nx1
        p = -((y-mu).square() + var) / self.scale().square()
        p -= np.log(2.0 * np.pi)
        p -= 2.0*self.scale().log()
        return 0.5*p.sum()  # sum over N

    def conditional_mean(self, X, f):
        return f

    def conditional_sample(self, X, f):
        return torch.distributions.normal.Normal(f, scale=self.scale()).sample()

    def predict(self, X, mu, var, ci=None, sigma=None, n=10000):
        if ci is None and sigma is None:
            return mu

        if self.output_dims is not None:
            var = self.scale()**2
            r = self._channel_indices(X)
            lower = torch.empty(mu.shape, device=config.device, dtype=config.dtype)
            upper = torch.empty(mu.shape, device=config.device, dtype=config.dtype)
            for i in range(self.output_dims):
                if sigma is None:
                    ci = torch.tensor(ci, device=config.device, dtype=config.dtype)
                    lower[r[i],:] = mu[r[i],:] + torch.sqrt(2.0*var[i])*self.scale()[i]*torch.erfinv(2.0*ci[0] - 1.0)
                    upper[r[i],:] = mu[r[i],:] + torch.sqrt(2.0)*self.scale()[i]*torch.erfinv(2.0*ci[1] - 1.0)
                else:
                    lower[r[i],:] = mu[r[i],:] - sigma*self.scale()[i]
                    upper[r[i],:] = mu[r[i],:] + sigma*self.scale()[i]
            return mu, lower, upper  # Nx1

        var += self.scale()**2
        if sigma is None:
            ci = torch.tensor(ci, device=config.device, dtype=config.dtype)
            lower = mu + torch.sqrt(2.0*var)*torch.erfinv(2.0*ci[0] - 1.0)
            upper = mu + torch.sqrt(2.0*var)*torch.erfinv(2.0*ci[1] - 1.0)
        else:
            lower = mu - sigma*var.sqrt()
            upper = mu + sigma*var.sqrt()
        return mu, lower, upper

class StudentTLikelihood(Likelihood):
    """
    Student's t likelihood given by

    $$ p(y|f) = \\frac{\\Gamma\\left(\\frac{\\nu+1}{2}\\right)}{\\Gamma\\left(\\frac{\\nu}{2}\\right)\\sqrt{\\pi\\nu}\\sigma} \\left( 1 + \\frac{(y-f)^2}{\\nu\\sigma^2} \\right)^{-(\\nu+1)/2} $$

    with \\(\\nu\\) the degrees of freedom and \\(\\sigma\\) the scale.

    Args:
        dof (float): Degrees of freedom.
        scale (float): Scale.
        quadratures (int): Number of quadrature points to use when approximating using Gauss-Hermite quadratures.

    Attributes:
        scale (mogptk.gpr.parameter.Parameter): Scale \\(\\sigma\\).
    """
    def __init__(self, dof=3, scale=1.0, quadratures=20):
        super().__init__(quadratures)
        self.dof = torch.tensor(dof, device=config.device, dtype=config.dtype)
        self.scale = Parameter(scale, lower=config.positive_minimum)

    def log_prob(self, X, y, f):
        # y: Nx1
        # f: NxQ
        p = -0.5 * (self.dof+1.0)*torch.log1p(((y-f)/self.scale()).square()/self.dof)
        p += torch.lgamma((self.dof+1.0)/2.0)
        p -= torch.lgamma(self.dof/2.0)
        p -= 0.5 * (torch.log(self.dof) + np.log(np.pi) + self.scale().square().log())
        return p  # NxQ

    def conditional_mean(self, X, f):
        if self.dof <= 1.0:
            return torch.full(f.shape, np.nan, device=config.device, dtype=config.dtype)
        return f

    # TODO: implement predict for certain values of dof

    def conditional_sample(self, X, f):
        return torch.distributions.studentT.StudentT(self.dof, f, self.scale()).sample()

class ExponentialLikelihood(Likelihood):
    """
    Exponential likelihood given by

    $$ p(y|f) = 1/h(f) e^{-y/h(f)} $$

    with \\(h\\) the link function and \\(y \\in [0.0,\\infty)\\).

    Args:
        link (function): Link function to map function values to the support of the likelihood.
        quadratures (int): Number of quadrature points to use when approximating using Gauss-Hermite quadratures.
    """
    def __init__(self, link=exp, quadratures=20):
        super().__init__(quadratures)
        self.link = link

    def validate_y(self, X, y):
        if torch.any(y < 0.0):
            raise ValueError("y must be positive")

    def log_prob(self, X, y, f):
        # y: Nx1
        # f: NxQ
        if self.link == exp:
            p = -y/self.link(f) - f
        else:
            p = -y/self.link(f) - self.link(f).log()
        return p  # NxQ

    def variational_expectation(self, X, y, mu, var):
        # y,mu,var: Nx1
        if self.link != exp:
            super().variational_expectation(X, y, mu, var)

        p = -mu - y * torch.exp(var/2.0 - mu)
        return p.sum()

    def conditional_mean(self, X, f):
        return self.link(f)

    #TODO: implement predict?

    def conditional_sample(self, X, f):
        if self.link != exp:
            raise ValueError("only exponential link function is supported")
        rate = 1.0/self.link(f)
        return torch.distributions.exponential.Exponential(rate).sample().log()

class LaplaceLikelihood(Likelihood):
    """
    Laplace likelihood given by

    $$ p(y|f) = \\frac{1}{2\\sigma}e^{-\\frac{|y-f|}{\\sigma}} $$

    with \\(\\sigma\\) the scale.

    Args:
        scale (float): Scale.
        quadratures (int): Number of quadrature points to use when approximating using Gauss-Hermite quadratures.

    Attributes:
        scale (mogptk.gpr.parameter.Parameter): Scale \\(\\sigma\\).
    """
    def __init__(self, scale=1.0, quadratures=20):
        super().__init__(quadratures)
        self.scale = Parameter(scale, lower=config.positive_minimum)

    def log_prob(self, X, y, f):
        # y: Nx1
        # f: NxQ
        p = -torch.log(2.0*self.scale()) - (y-f).abs()/self.scale()
        return p  # NxQ

    def conditional_mean(self, X, f):
        return f

    #def variational_expectation(self, X, y, mu, var):
    #    # TODO: doesn't seem right
    #    # y,mu,var: Nx1
    #    p = -y/self.scale() + mu/self.scale() - torch.log(2.0*self.scale())
    #    p += torch.sqrt(2.0*var/np.pi)/self.scale()
    #    return p.sum()

    #TODO: implement predict

    def conditional_sample(self, X, f):
        return torch.distributions.laplace.Laplace(f, self.scale()).sample()

class BernoulliLikelihood(Likelihood):
    """
    Bernoulli likelihood given by

    $$ p(y|f) = h(f)^k (1-h(f))^{n-k} $$

    with \\(h\\) the link function, \\(k\\) the number of \\(y\\) values equal to 1, \\(n\\) the number of data points, and \\(y \\in \\{0.0,1.0\\}\\).

    Args:
        link (function): Link function to map function values to the support of the likelihood.
        quadratures (int): Number of quadrature points to use when approximating using Gauss-Hermite quadratures.
    """
    def __init__(self, link=inv_probit):
        super().__init__()
        self.link = link

    def validate_y(self, X, y):
        if torch.any((y != 0.0) & (y != 1.0)):
            raise ValueError("y must have only 0.0 and 1.0 values")

    def log_prob(self, X, y, f):
        # y: Nx1
        # f: NxQ
        p = self.link(f)
        return torch.log(torch.where(0.5 <= y, p, 1.0-p))  # NxQ

    def conditional_mean(self, X, f):
        return self.link(f)

    def conditional_sample(self, X, f):
        return None

    def predict(self, X, mu, var, ci=None, sigma=None, n=10000):
        if self.link != inv_probit:
            return super().predict(X, mu, var, ci=ci, sigma=sigma, n=n)

        p = self.link(mu / torch.sqrt(1.0 + var))
        if ci is None and sigma is None:
            return p
        return p, p, p

class BetaLikelihood(Likelihood):
    """
    Beta likelihood given by

    $$ p(y|f) = \\frac{\\Gamma(\\sigma)}{\\Gamma\\left(h(f)\\sigma\\right)\\Gamma\\left((1-h(f)\\sigma\\right)} y^{h(f)\\sigma} (1-y)^{(1-h(f))\\sigma} $$

    with \\(h\\) the link function, \\(\\sigma\\) the scale, and \\(y \\in (0.0,1.0)\\).

    Args:
        scale (float): Scale.
        link (function): Link function to map function values to the support of the likelihood.
        quadratures (int): Number of quadrature points to use when approximating using Gauss-Hermite quadratures.

    Attributes:
        scale (mogptk.gpr.parameter.Parameter): Scale \\(\\sigma\\).
    """
    def __init__(self, scale=1.0, link=inv_probit, quadratures=20):
        super().__init__(quadratures)
        self.link = link
        self.scale = Parameter(scale, lower=config.positive_minimum)

    def validate_y(self, X, y):
        if torch.any((y <= 0.0) | (1.0 <= y)):
            raise ValueError("y must be in the range (0.0,1.0)")

    def log_prob(self, X, y, f):
        # y: Nx1
        # f: NxQ
        mixture = self.link(f)
        alpha = mixture * self.scale()
        beta = self.scale() - alpha

        p = (alpha-1.0)*y.log()
        p += (beta-1.0)*torch.log1p(-y)
        p += torch.lgamma(alpha+beta)
        p -= torch.lgamma(alpha)
        p -= torch.lgamma(beta)
        return p  # NxQ

    def conditional_mean(self, X, f):
        return self.link(f)

    def conditional_sample(self, X, f):
        if self.link != inv_probit:
            raise ValueError("only inverse probit link function is supported")
        mixture = self.link(f)
        alpha = mixture * self.scale()
        beta = self.scale() - alpha
        return probit(torch.distributions.beta.Beta(alpha, beta).sample())

class GammaLikelihood(Likelihood):
    """
    Gamma likelihood given by

    $$ p(y|f) = \\frac{1}{\\Gamma(k)h(f)^k} y^{k-1} e^{-y/h(f)} $$

    with \\(h\\) the link function, \\(k\\) the shape, and \\(y \\in (0.0,\\infty)\\). 

    Args:
        shape (float): Shape.
        link (function): Link function to map function values to the support of the likelihood.
        quadratures (int): Number of quadrature points to use when approximating using Gauss-Hermite quadratures.

    Attributes:
        shape (mogptk.gpr.parameter.Parameter): Shape \\(k\\).
    """
    def __init__(self, shape=1.0, link=exp, quadratures=20):
        super().__init__(quadratures)
        self.link = link
        self.shape = Parameter(shape, lower=config.positive_minimum)

    def validate_y(self, X, y):
        if torch.any(y <= 0.0):
            raise ValueError("y must be in the range (0.0,inf)")

    def log_prob(self, X, y, f):
        # y: Nx1
        # f: NxQ
        p = -y/self.link(f)
        p += (self.shape()-1.0)*y.log()
        p -= torch.lgamma(self.shape())
        if self.link == exp:
            p -= self.shape()*f
        else:
            p -= self.shape()*self.link(f).log()
        return p  # NxQ

    def variational_expectation(self, X, y, mu, var):
        # y,mu,var: Nx1
        if self.link != exp:
            super().variational_expectation(X, y, mu, var)

        p = -self.shape()*mu
        p -= torch.lgamma(self.shape())
        p += (self.shape() - 1.0) * y.log()
        p -= y * torch.exp(var/2.0 - mu)
        return p.sum()

    def conditional_mean(self, X, f):
        return self.shape()*self.link(f)

    def conditional_sample(self, X, f):
        if self.link != exp:
            raise ValueError("only exponential link function is supported")
        rate = 1.0/self.link(f)
        return torch.distributions.gamma.Gamma(self.shape(), rate).sample().log()

class PoissonLikelihood(Likelihood):
    """
    Poisson likelihood given by

    $$ p(y|f) = \\frac{1}{y!} h(f)^y e^{-h(f)} $$

    with \\(h\\) the link function and \\(y \\in \\mathbb{N}_0\\).

    Args:
        link (function): Link function to map function values to the support of the likelihood.
        quadratures (int): Number of quadrature points to use when approximating using Gauss-Hermite quadratures.
    """
    def __init__(self, link=exp, quadratures=20):
        super().__init__(quadratures)
        self.link = link

    def validate_y(self, X, y):
        if torch.any(y < 0.0):
            raise ValueError("y must be in the range [0.0,inf)")
        if not torch.all(y == y.long()):
            raise ValueError("y must have integer count values")

    def log_prob(self, X, y, f):
        # y: Nx1
        # f: NxQ
        if self.link == exp:
            p = y*f
        else:
            p = y*self.link(f).log()
        p -= torch.lgamma(y+1.0)
        p -= self.link(f)
        return p  # NxQ

    def variational_expectation(self, X, y, mu, var):
        # y,mu,var: Nx1
        if self.link != exp:
            super().variational_expectation(X, y, mu, var)

        p = y*mu - torch.exp(var/2.0 + mu) - torch.lgamma(y+1.0)
        return p.sum()

    def conditional_mean(self, X, f):
        return self.link(f)

    def conditional_sample(self, X, f):
        if self.link != exp:
            raise ValueError("only exponential link function is supported")
        rate = self.link(f)
        return torch.distributions.poisson.Poisson(rate).sample().log()

class WeibullLikelihood(Likelihood):
    """
    Weibull likelihood given by

    $$ p(y|f) = \\frac{k}{h(f)} \\left( \\frac{y}{h(f)} \\right)^{k-1} e^{-(y/h(f))^k} $$

    with \\(h\\) the link function, \\(k\\) the shape, and \\(y \\in (0.0,\\infty)\\).

    Args:
        shape (float): Shape.
        link (function): Link function to map function values to the support of the likelihood.
        quadratures (int): Number of quadrature points to use when approximating using Gauss-Hermite quadratures.

    Attributes:
        shape (mogptk.gpr.parameter.Parameter): Shape \\(k\\).
    """
    def __init__(self, shape=1.0, link=exp, quadratures=20):
        super().__init__(quadratures)
        self.link = link
        self.shape = Parameter(shape, lower=config.positive_minimum)

    def validate_y(self, X, y):
        if torch.any(y <= 0.0):
            raise ValueError("y must be in the range (0.0,inf)")

    def log_prob(self, X, y, f):
        # y: Nx1
        # f: NxQ
        if self.link == exp:
            p = -self.shape()*f
        else:
            p = -self.shape()*self.link(f).log()
        p += self.shape().log() + (self.shape()-1.0)*y.log()
        p -= (y/self.link(f))**self.shape()
        return p  # NxQ

    def conditional_mean(self, X, f):
        return self.link(f) * torch.lgamma(1.0 + 1.0/self.shape()).exp()

    def conditional_sample(self, X, f):
        if self.link != exp:
            raise ValueError("only exponential link function is supported")
        scale = self.link(f)
        return torch.distributions.weibull.Weibull(scale, self.shape()).sample().log()

class LogLogisticLikelihood(Likelihood):
    """
    Log-logistic likelihood given by

    $$ p(y|f) = \\frac{(k/h(f)) (y/h(f))^{k-1}}{\\left(1 + (y/h(f))^k\\right)^2} $$

    with \\(h\\) the link function, \\(k\\) the shape, and \\(y \\in (0.0,\\infty)\\).

    Args:
        shape (float): Shape.
        link (function): Link function to map function values to the support of the likelihood.
        quadratures (int): Number of quadrature points to use when approximating using Gauss-Hermite quadratures.

    Attributes:
        shape (mogptk.gpr.parameter.Parameter): Shape \\(k\\).
    """
    def __init__(self, shape=1.0, link=exp, quadratures=20):
        super().__init__(quadratures)
        self.link = link
        self.shape = Parameter(shape, lower=config.positive_minimum)

    def validate_y(self, X, y):
        if torch.any(y < 0.0):
            raise ValueError("y must be in the range [0.0,inf)")

    def log_prob(self, X, y, f):
        # y: Nx1
        # f: NxQ
        if self.link == exp:
            p = -self.shape()*f
        else:
            p = -self.shape()*self.link(f).log()
        p -= 2.0*torch.log1p((y/self.link(f))**self.shape())
        p += self.shape().log()
        p += (self.shape()-1.0)*y.log()
        return p  # NxQ

    def conditional_mean(self, X, f):
        return self.link(f) / torch.sinc(1.0/self.shape())

    def conditional_sample(self, X, f):
        if self.link != exp:
            raise ValueError("only exponential link function is supported")
        return log_logistic_distribution(f, 1.0/self.shape()).sample().log()

class LogGaussianLikelihood(Likelihood):
    """
    Log-Gaussian likelihood given by

    $$ p(y|f) = \\frac{1}{y\\sqrt{2\\pi\\sigma^2}} e^{-\\frac{(\\log(y) - f)^2}{2\\sigma^2}} $$

    with \\(\\sigma\\) the scale and \\(y \\in (0.0,\\infty)\\).

    Args:
        scale (float): Scale.
        quadratures (int): Number of quadrature points to use when approximating using Gauss-Hermite quadratures.

    Attributes:
        scale (mogptk.gpr.parameter.Parameter): Scale \\(\\sigma\\).
    """
    def __init__(self, scale=1.0, quadratures=20):
        super().__init__(quadratures)
        self.scale = Parameter(scale, lower=config.positive_minimum)

    def validate_y(self, X, y):
        if torch.any(y <= 0.0):
            raise ValueError("y must be in the range (0.0,inf)")

    def log_prob(self, X, y, f):
        # y: Nx1
        # f: NxQ
        logy = y.log()
        p = -0.5 * (np.log(2.0*np.pi) + 2.0*self.scale().log() + ((logy-f)/self.scale()).square())
        p -= logy
        return p  # NxQ

    def conditional_mean(self, X, f):
        return torch.exp(f + 0.5*self.scale().square())

    #TODO: implement variational_expectation?
    #TODO: implement predict

    def conditional_sample(self, X, f):
        return torch.distributions.log_normal.LogNormal(f, self.scale()).sample().log()

class ChiSquaredLikelihood(Likelihood):
    """
    Chi-squared likelihood given by

    $$ p(y|f) = \\frac{1}{2^{f/2}\\Gamma(f/2)} y^{f/2-1} e^{-y/2} $$

    with \\(y \\in (0.0,\\infty)\\).

    Args:
        link (function): Link function to map function values to the support of the likelihood.
        quadratures (int): Number of quadrature points to use when approximating using Gauss-Hermite quadratures.
    """
    def __init__(self, link=exp, quadratures=20):
        super().__init__(quadratures)
        self.link = link

    def validate_y(self, X, y):
        if torch.any(y <= 0.0):
            raise ValueError("y must be in the range (0.0,inf)")

    def log_prob(self, X, y, f):
        # y: Nx1
        # f: NxQ
        f = self.link(f)
        p = -0.5*f*np.log(2.0) - torch.lgamma(f/2.0) + (f/2.0-1.0)*y.log() - 0.5*y
        return p  # NxQ

    def conditional_mean(self, X, f):
        return self.link(f)

    def conditional_sample(self, X, f):
        if self.link != exp:
            raise ValueError("only exponential link function is supported")
        return torch.distributions.chi2.Chi2(self.link(f)).sample().log()

