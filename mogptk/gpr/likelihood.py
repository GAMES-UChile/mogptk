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

    def predict(self, mu, var, full=False):
        # TODO: implement quadrature approximation
        raise NotImplementedError()

    def log_prob(self, y, f):
        # log(p(y|f)), where p(y|f) is our likelihood
        # y: Nx1
        # f: NxM
        raise NotImplementedError()

    def variational_expectation(self, y, mu, var):
        # ∫ log(p(y|f)) q(f) df, where q(f) ~ N(mu,var) and p(y|f) is our likelihood
        # y,mu,var: Nx1
        # t: Mx1
        q = self.quadrature(lambda t: self.log_prob(y, mu + var.sqrt().mm(t.T)))  # Nx1
        return q.sum()  # sum over N

class GaussianLikelihood(Likelihood):
    def __init__(self, variance=1.0, name="GaussianLikelihood", quadratures=20):
        super(GaussianLikelihood, self).__init__(name, quadratures)

        self.variance = Parameter(variance, name="variance", lower=config.positive_minimum)

    def predict(self, mu, var, full=False):
        if full:
            return mu, var + self.variance()*torch.eye(var.shape[0])
        else:
            return mu, var + self.variance()

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

class StudentTLikelihood(Likelihood):
    def __init__(self, dof, scale=1.0, name="StudentTLikelihood", quadratures=20):
        super(StudentTLikelihood, self).__init__(name, quadratures)

        self.dof = dof
        self.scale = Parameter(scale, name="scale", lower=config.positive_minimum)

    def mean(self, f):
        return f

    def variance(self, f):
        return self.scale()**2 * self.dof / (self.dof-2.0)

    def log_prob(self, y, f):
        # y: Nx1
        # f: NxM
        p = torch.lgamma((self.dof+1.0)/2.0)
        p -= 0.5 * torch.log(self.dof * np.pi)
        p -= torch.log(self.scale())
        p -= torch.lgamma(self.dof/2.0)
        p -= 0.5 * (self.dof+1.0)*torch.log(1.0 + ((y-f)/self.scale()).square()/self.dof)
        return p  # NxM

class LaplaceLikelihood(Likelihood):
    def __init__(self, scale=1.0, name="LaplaceLikelihood", quadratures=20):
        super(LaplaceLikelihood, self).__init__(name, quadratures)

        self.scale = Parameter(scale, name="scale", lower=config.positive_minimum)

    def log_prob(self, y, f):
        # y: Nx1
        # f: NxM
        p = -torch.log(2.0*self.scale()) - (y-f).abs()/self.scale()
        return p  # NxM

    def variational_expectation(self, y, mu, var):
        # y,mu,var: Nx1
        p = -0.5 * ((y-mu).square() + var) / self.variance()
        p -= 0.5 * np.log(2.0 * np.pi)
        p -= 0.5 * self.variance().log()
        print("Laplace:", super().variational_expectation(y, mu, var), p.sum())
        return p.sum()  # sum over N
    

# TODO: implement log_prob: Beta, Softmax
# TODO: implement log_prob and var_exp: Bernoulli, Gamma, Laplace, Poisson
