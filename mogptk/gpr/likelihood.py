import torch
import numpy as np
from . import config, Parameter

class GaussHermiteQuadrature:
    def __init__(self, deg=20):
        t, w = np.polynomial.hermite.hermgauss(deg)
        t = t.reshape(-1,1)
        w = w.reshape(-1,1)
        self.t = torch.tensor(t*np.sqrt(2), device=config.device, dtype=config.dtype)  # Mx1
        self.w = torch.tensor(w/np.sqrt(np.pi), device=config.device, dtype=config.dtype)  # Mx1
        self.deg = deg

    def __call__(self, mu, var, F):
        return F(mu + var.sqrt().mm(self.t.T)).mm(self.w)  # Nx1

class Likelihood:
    def __init__(self, name, quadratures=20):
        self.name = name
        self.quadrature = GaussHermiteQuadrature(deg=quadratures)

    def log_prob(self, y, f):
        # log(p(y|f)), where p(y|f) is our likelihood
        # y: Nx1
        # f: NxM
        raise NotImplementedError()

    def variational_expectation(self, y, mu, var):
        # ∫ log(p(y|f)) q(f) df, where q(f) ~ N(mu,var) and p(y|f) is our likelihood
        # y,mu,var: Nx1
        q = self.quadrature(mu, var, lambda f: self.log_prob(y,f))  # Nx1
        return q.sum()  # sum over N

    #def mean(self, f):
    #    # f: NxM
    #    raise NotImplementedError()

    #def variance(self, f):
    #    # f: NxM
    #    raise NotImplementedError()

    def predict(self, mu, var, full=False):
        # mu,var: Nx1
        raise NotImplementedError()
        # ∫∫ y p(y|f) q(f) df dy,  ∫∫ y^2 p(y|f) q(f) df dy - (∫∫ y p(y|f) q(f) df dy)^2
        # where q(f) ~ N(mu,var) and p(y|f) is our likelihood
        # mu,var: Nx1
        #Ey = self.quadrature(mu, var, self.mean)
        #Eyy = self.quadrature(mu, var, lambda f: self.mean(f).square() + self.variance(f))
        #return Ey, Eyy-Ey**2 

class GaussianLikelihood(Likelihood):
    def __init__(self, variance=1.0, name="Gaussian", quadratures=20):
        super().__init__(name, quadratures)

        self.sigma = Parameter(variance, name="variance", lower=config.positive_minimum)

    def log_prob(self, y, f):
        # y: Nx1
        # f: NxM
        p = -0.5 * (np.log(2.0 * np.pi) + self.sigma().log() + (y-f).square()/self.sigma())
        return p  # NxM

    def variational_expectation(self, y, mu, var):
        # y,mu,var: Nx1
        p = -((y-mu).square() + var) / self.sigma()
        p -= np.log(2.0 * np.pi)
        p -= self.sigma().log()
        return 0.5*p.sum()  # sum over N

    def predict(self, mu, var, full=False):
        if full:
            return mu, var + self.sigma()*torch.eye(var.shape[0])
        else:
            return mu, var + self.sigma()

class StudentTLikelihood(Likelihood):
    def __init__(self, dof, scale=1.0, name="StudentT", quadratures=20):
        super().__init__(name, quadratures)

        self.dof = torch.tensor(dof, device=config.device, dtype=config.dtype)
        self.scale = Parameter(scale, name="scale", lower=config.positive_minimum)

    def log_prob(self, y, f):
        # y: Nx1
        # f: NxM
        p = 0.5 * (self.dof+1.0)*torch.log(1.0 + ((y-f)/self.scale()).square()/self.dof)
        p += torch.lgamma((self.dof+1.0)/2.0)
        p -= torch.lgamma(self.dof/2.0)
        p -= 0.5 * torch.log(self.dof * np.pi)
        p -= torch.log(self.scale())
        return p  # NxM

    #def mean(self, f):
    #    return f

    #def variance(self, f):
    #    if self.dof < 2.0:
    #        var = np.nan
    #    else:
    #        var = self.scale()**2 * self.dof/(self.dof-2.0)
    #    return torch.full(f.shape, var, device=config.device, dtype=config.dtype)

class LaplaceLikelihood(Likelihood):
    def __init__(self, scale=1.0, name="Laplace", quadratures=20):
        super().__init__(name, quadratures)

        self.scale = Parameter(scale, name="scale", lower=config.positive_minimum)

    def log_prob(self, y, f):
        # y: Nx1
        # f: NxM
        p = -torch.log(2.0*self.scale()) - (y-f).abs()/self.scale()
        return p  # NxM

def inv_probit(x):
    jitter = 1e-3
    return 0.5*(1.0+torch.erf(x/np.sqrt(2.0))) * (1.0-2.0*jitter) + jitter

def logistic(x):
    return 1.0/(1.0+torch.exp(-x))

class BernoulliLikelihood(Likelihood):
    def __init__(self, scale=1.0, link=inv_probit, name="Bernoulli", quadratures=20):
        super().__init__(name, quadratures)

        self.link = link

    def log_prob(self, y, f):
        # y: Nx1
        # f: NxM
        p = self.link(f)
        return torch.log(torch.where(y == 1.0, p, 1.0-p))  # NxM

    def predict(self, mu, var, full=False):
        if self.link == inv_probit:
            p = self.link(mu / torch.sqrt(1.0 + var))
            return p, torch.zeros(var.shape)
        else:
            p = self.quadrature(mu, var, self.link)
            return p, torch.zeros(var.shape)

# TODO: implement log_prob: Beta, Softmax
# TODO: implement log_prob and var_exp: Gamma, Laplace, Poisson
