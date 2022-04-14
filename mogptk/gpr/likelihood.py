import torch
import numpy as np
from . import config, Parameter

def exp(x):
    return torch.exp(x)

def inv_probit(x):
    jitter = 1e-3
    return 0.5*(1.0+torch.erf(x/np.sqrt(2.0))) * (1.0-2.0*jitter) + jitter

# also inv_logit or sigmoid
def logistic(x):
    return 1.0/(1.0+torch.exp(-x))

class GaussHermiteQuadrature:
    def __init__(self, deg=20, t_scale=None, w_scale=None):
        t, w = np.polynomial.hermite.hermgauss(deg)
        t = t.reshape(-1,1)
        w = w.reshape(-1,1)
        if t_scale is not None:
            t *= t_scale
        if w_scale is not None:
            w *= w_scale
        self.t = torch.tensor(t, device=config.device, dtype=config.dtype)  # Mx1
        self.w = torch.tensor(w, device=config.device, dtype=config.dtype)  # Mx1
        self.deg = deg

    def __call__(self, mu, var, F):
        return F(mu + var.sqrt().mm(self.t.T)).mm(self.w)  # Nx1

class Likelihood:
    def __init__(self, name="Likelihood", quadratures=20):
        self.name = name
        self.quadrature = GaussHermiteQuadrature(deg=quadratures, t_scale=np.sqrt(2), w_scale=1.0/np.sqrt(np.pi))
        self.output_dims = 1

    def validate_y(self, y):
        pass

    def log_prob(self, y, f, X=None):
        # log(p(y|f)), where p(y|f) is our likelihood
        # y: Nx1
        # f: NxM
        raise NotImplementedError()

    def variational_expectation(self, y, mu, var, X=None):
        # ∫ log(p(y|f)) q(f) df, where q(f) ~ N(mu,var) and p(y|f) is our likelihood
        # y,mu,var: Nx1
        q = self.quadrature(mu, var, lambda f: self.log_prob(y,f,X=X))  # Nx1
        return q.sum()  # sum over N

    def predictive_y(self, f, X=None):
        # f: NxM
        raise NotImplementedError()

    def predictive_yy(self, f, X=None):
        # f: NxM
        raise NotImplementedError()

    def predict(self, mu, var, X=None, full=False):
        # ∫∫ y p(y|f) q(f) df dy,  ∫∫ y^2 p(y|f) q(f) df dy - (∫∫ y p(y|f) q(f) df dy)^2
        # where q(f) ~ N(mu,var) and p(y|f) is our likelihood
        # mu,var: Nx1
        Ey = self.quadrature(mu, var, lambda f: self.predictive_y(f,X=X))
        Eyy = self.quadrature(mu, var, lambda f: self.predictive_yy(f,X=X))
        return Ey, Eyy-Ey**2 

class MultiOutputLikelihood(Likelihood):
    def __init__(self, *likelihoods, name="MultiOutputLikelihood", quadratures=20):
        super().__init__(name=name, quadratures=quadratures)

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
        self.likelihoods = likelihoods

    def _channel_indices(self, X):
        c = X[:,0].long()
        m = [c==j for j in range(self.output_dims)]
        r = [torch.nonzero(m[j], as_tuple=False).reshape(-1) for j in range(self.output_dims)]  # as_tuple avoids warning
        return r

    def validate_y(self, y):
        for likelihood in likelihoods:
            likelihood.validate_y(y)

    def log_prob(self, y, f, X=None):
        # y: Nx1
        # f: NxM
        if self.output_dims == 1:
            return self.likelihoods[0].log_prob(y,f)

        r = self._channel_indices(X)
        res = torch.empty(f.shape, device=config.device, dtype=config.dtype)
        for i in range(self.output_dims):
            res[r[i],:] = self.likelihoods[i].log_prob(y[r[i],:], f[r[i],:])
        return res  # NxM

    def variational_expectation(self, y, mu, var, X=None):
        # y,mu,var: Nx1
        if self.output_dims == 1:
            return self.likelihoods[0].variational_expectation(y,mu,var)

        q = torch.tensor(0.0, dtype=config.dtype, device=config.device)
        r = self._channel_indices(X)
        for i in range(self.output_dims):
            q += self.likelihoods[i].variational_expectation(y[r[i],:], mu[r[i],:], var[r[i],:]).sum()  # sum over N
        return q

    def predictive_y(self, f, X=None):
        # f: NxM
        if self.output_dims == 1:
            return self.likelihoods[0].predictive_y(f)

        r = self._channel_indices(X)
        res = torch.empty(f.shape, device=config.device, dtype=config.dtype)
        for i in range(self.output_dims):
            res[r[i],:] = self.likelihoods[i].predictive_y(f[r[i],:])
        return res  # NxM

    def predictive_yy(self, f, X=None):
        # f: NxM
        if self.output_dims == 1:
            return self.likelihoods[0].predictive_yy2(f)

        r = self._channel_indices(X)
        res = torch.empty(f.shape, device=config.device, dtype=config.dtype)
        for i in range(self.output_dims):
            res[r[i],:] = self.likelihoods[i].predictive_yy(f[r[i],:])
        return res  # NxM

    # TODO: predict is not possible?
    #def predict(self, mu, var, X=None, full=False):
    #    # mu: Nx1
    #    # var: Nx1 or NxN
    #    if self.output_dims == 1:
    #        return self.likelihoods[0].predict(mu,var,full=full)

    #    r = self._channel_indices(X)
    #    Ey = torch.empty(mu.shape, device=config.device, dtype=config.dtype)
    #    Eyy = torch.empty(var.shape, device=config.device, dtype=config.dtype)
    #    if full:
    #        for i in range(self.output_dims):
    #            r1 = r[i].reshape(-1,1)
    #            r2 = r[i].reshape(1,-1)
    #            Ey[r[i],:], Eyy[r1,r2] = self.likelihoods[i].predict(mu[r[i],:], var[r1,r2], full=True)
    #    else:
    #        for i in range(self.output_dims):
    #            Ey[r[i],:], Eyy[r[i],:] = self.likelihoods[i].predict(mu[r[i],:], var[r[i],:], full=False)
    #    return Ey, Eyy-Ey.square()

class GaussianLikelihood(Likelihood):
    def __init__(self, variance=1.0, name="Gaussian", quadratures=20):
        super().__init__(name, quadratures)

        self.variance = Parameter(variance, name="variance", lower=config.positive_minimum)

    def log_prob(self, y, f, X=None):
        # y: Nx1
        # f: NxM
        p = -0.5 * (np.log(2.0 * np.pi) + self.variance().log() + (y-f).square()/self.variance())
        return p  # NxM

    def variational_expectation(self, y, mu, var, X=None):
        # y,mu,var: Nx1
        p = -((y-mu).square() + var) / self.variance()
        p -= np.log(2.0 * np.pi)
        p -= self.variance().log()
        return 0.5*p.sum()  # sum over N

    def predictive_y(self, f, X=None):
        return f

    def predictive_yy(self, f, X=None):
        return f.square() + self.variance()

    def predict(self, mu, var, X=None, full=False):
        if full:
            return mu, var + self.variance()*torch.eye(var.shape[0])
        else:
            return mu, var + self.variance()

class StudentTLikelihood(Likelihood):
    def __init__(self, dof=3, scale=1.0, name="StudentT", quadratures=20):
        super().__init__(name, quadratures)

        self.dof = torch.tensor(dof, device=config.device, dtype=config.dtype)
        self.scale = Parameter(scale, name="scale", lower=config.positive_minimum)

    def log_prob(self, y, f, X=None):
        # y: Nx1
        # f: NxM
        p = -0.5 * (self.dof+1.0)*torch.log(1.0 + ((y-f)/self.scale()).square()/self.dof)
        p += torch.lgamma((self.dof+1.0)/2.0)
        p -= torch.lgamma(self.dof/2.0)
        p -= 0.5 * torch.log(self.dof*np.pi*self.scale()**2)
        return p  # NxM

    def predictive_y(self, f, X=None):
        if self.dof <= 1.0:
            return torch.full(f.shape, np.nan, device=config.device, dtype=config.dtype)
        return f

    def predictive_yy(self, f, X=None):
        if self.dof <= 2.0:
            return torch.full(f.shape, np.nan, device=config.device, dtype=config.dtype)
        return f.square() + self.scale()**2 * self.dof/(self.dof-2.0)

class ExponentialLikelihood(Likelihood):
    def __init__(self, link=exp, name="Exponential", quadratures=20):
        super().__init__(name, quadratures)

        self.link = link

    def log_prob(self, y, f, X=None):
        # y: Nx1
        # f: NxM
        p = -y/self.link(f) - torch.log(self.link(f))
        return p  # NxM

    def predictive_y(self, f, X=None):
        return self.link(f)

    def predictive_yy(self, f, X=None):
        return 2.0*self.link(f)**2

class LaplaceLikelihood(Likelihood):
    def __init__(self, scale=1.0, name="Laplace", quadratures=20):
        super().__init__(name, quadratures)

        self.scale = Parameter(scale, name="scale", lower=config.positive_minimum)

    def log_prob(self, y, f, X=None):
        # y: Nx1
        # f: NxM
        p = -torch.log(2.0*self.scale()) - (y-f).abs()/self.scale()
        return p  # NxM

    def predictive_y(self, f, X=None):
        return f

    def predictive_yy(self, f, X=None):
        return f.square() + 2.0*self.scale()**2

class BernoulliLikelihood(Likelihood):
    def __init__(self, scale=1.0, link=inv_probit, name="Bernoulli", quadratures=20):
        super().__init__(name, quadratures)

        self.link = link

    def validate_y(self, y):
        if torch.all((y == 0.0) | (y == 1.0))
            raise ValueError("y must have only 0.0 and 1.0")

    def log_prob(self, y, f, X=None):
        # y: Nx1
        # f: NxM
        p = self.link(f)
        return torch.log(torch.where(0.5 <= y, p, 1.0-p))  # NxM

    def predictive_y(self, f, X=None):
        return self.link(f)

    def predictive_yy(self, f, X=None):
        return self.link(f)

    def predict(self, mu, var, X=None, full=False):
        if self.link == inv_probit:
            p = self.link(mu / torch.sqrt(1.0 + var))
            if full:
                return p.diagonal().reshape(-1,1), p-p.square() # TODO: correct?
            return p, p-p.square()
        else:
            return super().predict(mu, var, X=X, full=full)

class BetaLikelihood(Likelihood):
    def __init__(self, scale=1.0, link=inv_probit, name="Beta", quadratures=20):
        super().__init__(name, quadratures)

        self.link = link
        self.scale = Parameter(scale, name="scale", lower=config.positive_minimum)

    def validate_y(self, y):
        if not torch.all((0.0 < y) & (y < 1.0))
            raise ValueError("y must be in the range (0.0,1.0)")

    def log_prob(self, y, f, X=None):
        # y: Nx1
        # f: NxM
        mixture = self.link(f)
        alpha = mixture * self.scale()
        beta = (1.0-mixture) * self.scale()

        p = (alpha-1.0)*torch.log(y)
        p += (beta-1.0)*torch.log(1.0-y)
        p += torch.lgamma(alpha+beta)
        p -= torch.lgamma(alpha)
        p -= torch.lgamma(beta)
        return p  # NxM

    def predictive_y(self, f, X=None):
        return self.link(f)

    def predictive_yy(self, f, X=None):
        mixture = self.link(f)
        return (mixture + mixture.square()*self.scale()) / (self.scale() + 1.0)

class GammaLikelihood(Likelihood):
    def __init__(self, shape=1.0, link=exp, name="Gamma", quadratures=20):
        super().__init__(name, quadratures)

        self.link = link
        self.shape = Parameter(shape, name="shape", lower=config.positive_minimum)

    def validate_y(self, y):
        if not torch.all(0.0 < y)
            raise ValueError("y must be in the range (0.0,inf)")

    def log_prob(self, y, f, X=None):
        # y: Nx1
        # f: NxM
        p = -y/self.link(f)
        p += (self.shape()-1.0)*torch.log(y)
        p -= torch.lgamma(self.shape())
        p -= self.shape()*torch.log(self.link(f))
        return p  # NxM

    def predictive_y(self, f, X=None):
        return self.shape()*self.link(f)

    def predictive_yy(self, f, X=None):
        t = self.shape()*self.link(f)
        return t + t.square()

# TODO: implement: Softmax and Poisson likelihoods?
