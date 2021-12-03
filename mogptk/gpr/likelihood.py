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

    def predictive_y(self, f):
        # f: NxM
        raise NotImplementedError()

    def predictive_yy2(self, f):
        # f: NxM
        raise NotImplementedError()

    def predict(self, mu, var, full=False):
        # ∫∫ y p(y|f) q(f) df dy,  ∫∫ y^2 p(y|f) q(f) df dy - (∫∫ y p(y|f) q(f) df dy)^2
        # where q(f) ~ N(mu,var) and p(y|f) is our likelihood
        # mu,var: Nx1
        Ey = self.quadrature(mu, var, self.predictive_y)
        Eyy = self.quadrature(mu, var, self.predictive_yy)
        return Ey, Eyy-Ey**2 

class GaussianLikelihood(Likelihood):
    def __init__(self, variance=1.0, name="Gaussian", quadratures=20):
        super().__init__(name, quadratures)

        self.variance = Parameter(variance, name="variance", lower=config.positive_minimum)

    def log_prob(self, y, f):
        # y: Nx1
        # f: NxM
        p = -0.5 * (np.log(2.0 * np.pi) + self.variance().log() + (y-f).square()/self.variance())
        return p  # NxM

    def variational_expectation(self, y, mu, var):
        # y,mu,var: Nx1
        p = -((y-mu).square() + var) / self.variance()
        p -= np.log(2.0 * np.pi)
        p -= self.variance().log()
        return 0.5*p.sum()  # sum over N

    def predictive_y(self, f):
        return f

    def predictive_yy(self, f):
        return f.square() + self.variance()

    def predict(self, mu, var, full=False):
        if full:
            return mu, var + self.variance()*torch.eye(var.shape[0])
        else:
            return mu, var + self.variance()

class StudentTLikelihood(Likelihood):
    def __init__(self, dof=3, scale=1.0, name="StudentT", quadratures=20):
        super().__init__(name, quadratures)

        self.dof = torch.tensor(dof, device=config.device, dtype=config.dtype)
        self.scale = Parameter(scale, name="scale", lower=config.positive_minimum)

    def log_prob(self, y, f):
        # y: Nx1
        # f: NxM
        p = -0.5 * (self.dof+1.0)*torch.log(1.0 + ((y-f)/self.scale()).square()/self.dof)
        p += torch.lgamma((self.dof+1.0)/2.0)
        p -= torch.lgamma(self.dof/2.0)
        p -= 0.5 * torch.log(self.dof*np.pi*self.scale()**2)
        return p  # NxM

    def predictive_y(self, f):
        if self.dof <= 1.0:
            return torch.full(f.shape, np.nan, device=config.device, dtype=config.dtype)
        return f

    def predictive_yy(self, f):
        if self.dof <= 2.0:
            return torch.full(f.shape, np.nan, device=config.device, dtype=config.dtype)
        return f.square() + self.scale()**2 * self.dof/(self.dof-2.0)

class LaplaceLikelihood(Likelihood):
    def __init__(self, scale=1.0, name="Laplace", quadratures=20):
        super().__init__(name, quadratures)

        self.scale = Parameter(scale, name="scale", lower=config.positive_minimum)

    def log_prob(self, y, f):
        # y: Nx1
        # f: NxM
        p = -torch.log(2.0*self.scale()) - (y-f).abs()/self.scale()
        return p  # NxM

    def predictive_y(self, f):
        return f

    def predictive_yy(self, f):
        return f.square() + 2.0*self.scale()**2

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
        return torch.log(torch.where(0.5 <= y, p, 1.0-p))  # NxM

    def predictive_y(self, f):
        return self.link(f)

    def predictive_yy(self, f):
        return self.link(f)

    def predict(self, mu, var, full=False):
        if self.link == inv_probit:
            p = self.link(mu / torch.sqrt(1.0 + var))
            return p, torch.zeros(var.shape)
        else:
            return super().predict(mu, var, full=full)

# TODO: implement log_prob: Beta, Softmax
# TODO: implement log_prob and var_exp: Gamma, Laplace, Poisson
