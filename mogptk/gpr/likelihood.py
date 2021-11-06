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

    def __call__(self, F):
        return F(self.t).mm(self.w)  # Nx1

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
        # t: Mx1
        q = self.quadrature(lambda t: self.log_prob(y, var.mm(t.T) + mu))  # Nx1
        return q.sum()  # sum over N

class GaussianLikelihood(Likelihood):
    def __init__(self, variance=1.0, name="GaussianLikelihood", quadratures=20):
        super(GaussianLikelihood, self).__init__(name, quadratures)

        self.variance = Parameter(variance, name="variance", lower=config.positive_minimum)

    def log_prob(self, y, f):
        # y: Nx1
        # f: NxM
        p = -0.5 * (np.log(2.0 * np.pi) + self.variance().log() + (y-f).square()/self.variance())
        return p  # NxM

    def variational_expectation(self, y, mu, var):
        # y,mu,var: Nx1
        p = -0.5 * ((y-mu).square() + var) / self.variance()
        p -= 0.5 * np.log(2.0 * np.pi)
        p -= 0.5 * self.variance().log()
        return p.sum()  # sum over N
