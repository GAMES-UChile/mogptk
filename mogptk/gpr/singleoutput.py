import torch
import numpy as np
from . import Kernel, Parameter, config

class ConstantKernel(Kernel):
    def __init__(self, input_dims=None, active_dims=None, name="Constant"):
        super().__init__(input_dims, active_dims, name)

        sigma = torch.rand(1)

        self.sigma = Parameter(sigma, lower=config.positive_minimum)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        if X2 is None:
            X2 = X1
        return self.sigma()**2 * torch.ones(X1.shape[0], X2.shape[0], dtype=X1.dtype, device=X1.device)

class WhiteKernel(Kernel):
    def __init__(self, input_dims=None, active_dims=None, name="White"):
        super().__init__(input_dims, active_dims, name)

        sigma = torch.rand(1)

        self.sigma = Parameter(sigma, lower=config.positive_minimum)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        if X2 is None:
            return self.sigma()**2 * torch.eye(X1.shape[0], X1.shape[0], dtype=X1.dtype, device=X1.device)
        return torch.zeros(X1.shape[0], X2.shape[0], dtype=X1.dtype, device=X1.device)

class LinearKernel(Kernel):
    def __init__(self, input_dims=None, active_dims=None, name="Linear"):
        super().__init__(input_dims, active_dims, name)

        constant = torch.rand(1)
        sigma = torch.rand(1)

        self.constant = Parameter(constant, lower=0.0)
        self.sigma = Parameter(sigma, lower=config.positive_minimum)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        if X2 is None:
            X2 = X1
        return self.sigma()**2 * X1.mm(X2.T) + self.constant()

class PolynomialKernel(Kernel):
    def __init__(self, degree, input_dims=None, active_dims=None, name="Polynomial"):
        super().__init__(input_dims, active_dims, name)

        offset = torch.rand(1)
        sigma = torch.rand(1)

        self.degree = degree
        self.offset = Parameter(offset, lower=0.0)
        self.sigma = Parameter(sigma, lower=config.positive_minimum)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        if X2 is None:
            X2 = X1
        return (self.sigma()**2 * X1.mm(X2.T) + self.offset())**self.degree

class PhiKernel(Kernel):
    def __init__(self, phi, input_dims, active_dims=None, name="Phi"):
        super().__init__(input_dims, active_dims, name)

        feature_dims = phi(torch.ones(input_dims,1)).shape[1]
        variance = torch.ones(feature_dims)

        self.phi = phi
        self.variance = Parameter(variance, lower=config.positive_minimum)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        if X2 is None:
            X = self.phi(X1)
            return X.mm(self.variance().diagflat().mm(X.T))
        else:
            return self.phi(X1).mm(self.variance().diagflat().mm(self.phi(X2).T))

class ExponentialKernel(Kernel):
    def __init__(self, input_dims, active_dims=None, name="Exponential"):
        super().__init__(input_dims, active_dims, name)

        l = torch.rand(input_dims)
        sigma = torch.rand(1)

        self.l = Parameter(l, lower=config.positive_minimum)
        self.sigma = Parameter(sigma, lower=config.positive_minimum)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        dist = torch.abs(self.distance(X1,X2))  # NxMxD
        exp = -0.5*torch.tensordot(dist, 1.0/self.l(), dims=1)  # NxM
        return self.sigma()**2 * torch.exp(exp)

class SquaredExponentialKernel(Kernel):
    def __init__(self, input_dims, active_dims=None, name="SE"):
        super().__init__(input_dims, active_dims, name)

        l = torch.rand(input_dims)
        sigma = torch.rand(1)

        self.l = Parameter(l, lower=config.positive_minimum)
        self.sigma = Parameter(sigma, lower=config.positive_minimum)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        sqdist = self.squared_distance(X1,X2)  # NxMxD
        exp = -0.5*torch.tensordot(sqdist, 1.0/self.l()**2, dims=1)  # NxM
        return self.sigma()**2 * torch.exp(exp)

class RationalQuadraticKernel(Kernel):
    def __init__(self, alpha, input_dims, active_dims=None, name="RQ"):
        super().__init__(input_dims, active_dims, name)

        l = torch.rand(input_dims)
        sigma = torch.rand(1)

        self.alpha = alpha
        self.l = Parameter(l, lower=config.positive_minimum)
        self.sigma = Parameter(sigma, lower=config.positive_minimum)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        sqdist = self.squared_distance(X1,X2)  # NxMxD
        power = 1.0+0.5*torch.tensordot(sqdist, 1.0/self.l()**2, dims=1)/self.alpha  # NxM
        return self.sigma()**2 * torch.pow(power,-self.alpha)

class PeriodicKernel(Kernel):
    def __init__(self, input_dims, active_dims=None, name="Periodic"):
        super().__init__(input_dims, active_dims, name)

        l = torch.rand(input_dims)
        p = torch.rand(1)
        sigma = torch.rand(1)

        self.l = Parameter(l, lower=config.positive_minimum)
        self.p = Parameter(p, lower=config.positive_minimum)
        self.sigma = Parameter(sigma, lower=config.positive_minimum)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        sin = torch.sin(np.pi * self.distance(X1,X2) / self.p())  # NxMxD
        exp = -2.0 * torch.tensordot(sin**2, 1.0/self.l()**2, dims=1)  # NxM
        return self.sigma()**2 * torch.exp(exp)

class LocallyPeriodicKernel(Kernel):
    def __init__(self, input_dims, active_dims=None, name="LocallyPeriodic"):
        super().__init__(input_dims, active_dims, name)

        l = torch.rand(input_dims)
        p = torch.rand(1)
        sigma = torch.rand(1)

        self.l = Parameter(l, lower=config.positive_minimum)
        self.p = Parameter(p, lower=config.positive_minimum)
        self.sigma = Parameter(sigma, lower=config.positive_minimum)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        sin = torch.sin(np.pi * self.distance(X1,X2) / self.p())  # NxMxD
        exp1 = -2.0 * torch.tensordot(sin**2, 1.0/self.l()**2, dims=1)  # NxM
        exp2 = -0.5 * torch.tensordot(self.squared_distance(X1,X2), 1.0/self.l()**2, dims=1)  # NxM
        return self.sigma()**2 * torch.exp(exp1) * torch.exp(exp2)

class CosineKernel(Kernel):
    def __init__(self, input_dims, active_dims=None, name="Cosine"):
        super().__init__(input_dims, active_dims, name)

        l = torch.rand(input_dims)
        sigma = torch.rand(1)

        self.l = Parameter(l, lower=config.positive_minimum)
        self.sigma = Parameter(sigma, lower=config.positive_minimum)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        cos = 2.0*np.pi * torch.tensordot(self.distance(X1,X2), 1.0/self.l(), dims=1)  # NxM
        return self.sigma()**2 * torch.cos(cos)

class SincKernel(Kernel):
    def __init__(self, input_dims, active_dims=None, name="Sinc"):
        super().__init__(input_dims, active_dims, name)

        self.bandwidth = 1.0#torch.rand(input_dims)
        frequency = torch.zeros(input_dims)
        sigma = torch.rand(1)

        self.frequency = Parameter(frequency, lower=config.positive_minimum)
        #self.bandwidth = Parameter(bandwidth, lower=config.positive_minimum)
        self.sigma = Parameter(sigma, lower=config.positive_minimum)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        tau = self.distance(X1,X2)  # NxMxD
        sinc = tau * self.bandwidth  # NxM
        cos = 2.0*np.pi * torch.tensordot(tau, self.frequency(), dims=1)  # NxM
        return self.sigma()**2 * torch.sinc(sinc) * torch.cos(cos)

class SpectralKernel(Kernel):
    def __init__(self, input_dims, active_dims=None, name="SM"):
        super().__init__(input_dims, active_dims, name)

        weight = torch.rand(1)
        mean = torch.rand(input_dims)
        variance = torch.ones(input_dims)

        self.weight = Parameter(weight, lower=config.positive_minimum)
        self.mean = Parameter(mean, lower=config.positive_minimum)
        self.variance = Parameter(variance, lower=config.positive_minimum)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        tau = self.distance(X1,X2)  # NxMxD
        exp = -2.0*np.pi**2 * tau**2 * self.variance().reshape(1,1,-1)  # NxMxD
        cos = 2.0*np.pi * tau * self.mean().reshape(1,1,-1)  # NxMxD
        return self.weight() * torch.prod(torch.exp(exp) * torch.cos(cos), dim=2)

class MaternKernel(Kernel):
    def __init__(self, nu=0.5, input_dims=None, active_dims=None, name="Matérn"):
        super().__init__(input_dims, active_dims, name)

        if nu not in [0.5, 1.5, 2.5]:
            raise ValueError("nu parameter must be 0.5, 1.5, or 2.5")

        l = torch.rand(input_dims)
        sigma = torch.rand(1)

        self.nu = nu
        self.l = Parameter(l, lower=1e-6)
        self.sigma = Parameter(sigma, lower=1e-6)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        if X2 is None:
            X2 = X1

        if self.nu == 0.5:
            constant = 1.0
        elif self.nu == 1.5:
            constant = 1.0 + np.sqrt(3.0)*dist
        elif self.nu == 2.5:
            constant = 1.0 + np.sqrt(5.0)*dist + 5.0/3.0*dist**2

        dist = torch.abs(self.distance(X1,X2))
        exp = -np.sqrt(self.nu*2.0) * torch.tensordot(dist, 1.0/self.l(), dims=1)
        return self.sigma()**2 * constant * torch.exp(exp)
