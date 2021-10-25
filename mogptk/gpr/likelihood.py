import torch
import numpy as np
from . import config, Parameter

class GaussHermiteQuadrature:
    def __init__(self, deg=20):
        x, w = np.polynomial.hermite.hermgauss(deg)
        x = x.reshape(-1,1)
        w = w.reshape(-1,1)
        self.x = torch.tensor(x*np.sqrt(2), device=config.device, dtype=config.dtype)  # Mx1
        self.w = torch.tensor(w/np.sqrt(np.pi), device=config.device, dtype=config.dtype)  # Mx1
        self.deg = deg

    def __call__(self, F):
        return F(self.x).mm(self.w)  # Nx1

class Likelihood:
    def __init__(self, name, quadratures=20):
        self.name = name
        self.quadrature = GaussHermiteQuadrature(deg=quadratures)
    
    def log_prob(self, y, f):
        raise NotImplementedError()

    def variational_expectation(self, y, mu, var):
        # y,mu,var: Nx1
        q = self.quadrature(lambda x: self.log_prob(y, var.mm(x.T) + mu))
        return q.sum()  # sum over N

class GaussianLikelihood(Likelihood):
    def __init__(self, variance=1.0, name="GaussianLikelihood", quadratures=20):
        super(GaussianLikelihood, self).__init__(name, quadratures)

        self.variance = Parameter(variance, name="variance", lower=config.positive_minimum)

    def log_prob(self, y, f):
        p = -0.5 * (np.log(2.0 * np.pi) + self.variance().log() + (y-f).square()/self.variance())
        return p

    def variational_expectation(self, y, mu, var):
        # y,mu,var: Nx1
        p = -0.5 * ((y-mu).square() + var) / self.variance()
        p -= 0.5 * np.log(2.0 * np.pi)
        p -= 0.5 * self.variance().log()
        return p.sum()  # sum over N
