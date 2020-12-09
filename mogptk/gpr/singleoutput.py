import torch
import numpy as np
from . import Kernel, Parameter, config

class LinearKernel(Kernel):
    def __init__(self, input_dims=None, active_dims=None, name="Linear"):
        super(LinearKernel, self).__init__(input_dims, active_dims, name)

        constant = torch.rand(1)

        self.constant = Parameter(constant, lower=0.0)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1,X2 = self._check_input(X1,X2)
        if X2 is None:
            X2 = X1

        return X1.mm(X2.T) + self.constant()

class PolynomialKernel(Kernel):
    def __init__(self, degree, input_dims=None, active_dims=None, name="Polynomial"):
        super(PolynomialKernel, self).__init__(input_dims, active_dims, name)

        offset = torch.rand(1)

        self.degree = degree
        self.offset = Parameter(offset, lower=0.0)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1,X2 = self._check_input(X1,X2)
        if X2 is None:
            X2 = X1

        return (X1.mm(X2.T) + self.offset())**self.degree

class PhiKernel(Kernel):
    def __init__(self, phi, input_dims, active_dims=None, name="Phi"):
        super(PhiKernel, self).__init__(input_dims, active_dims, name)

        feature_dims = phi(torch.ones(input_dims,1)).shape[1]
        variance = torch.ones(feature_dims)

        self.phi = phi
        self.variance = Parameter(variance, lower=config.positive_minimum)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1,X2 = self._check_input(X1,X2)

        if X2 is None:
            X = self.phi(X1)
            return X.mm(self.variance().diagflat().mm(X.T))
        else:
            return self.phi(X1).mm(self.variance().diagflat().mm(self.phi(X2).T))

class SquaredExponentialKernel(Kernel):
    def __init__(self, input_dims, active_dims=None, name="SE"):
        super(SquaredExponentialKernel, self).__init__(input_dims, active_dims, name)

        l = torch.rand(input_dims)
        sigma = torch.rand(1)

        self.l = Parameter(l, lower=config.positive_minimum)
        self.sigma = Parameter(sigma, lower=config.positive_minimum)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1,X2 = self._check_input(X1,X2)

        sqdist = self.squared_distance(X1,X2)  # NxMxD
        exp = torch.exp(-0.5*torch.tensordot(sqdist, 1.0/self.l()**2, dims=1))  # NxM
        return self.sigma()**2 * exp

class RationalQuadraticKernel(Kernel):
    def __init__(self, alpha, input_dims, active_dims=None, name="RQ"):
        super(RationalQuadraticKernel, self).__init__(input_dims, active_dims, name)

        l = torch.rand(input_dims)
        sigma = torch.rand(1)

        self.alpha = alpha
        self.l = Parameter(l, lower=config.positive_minimum)
        self.sigma = Parameter(sigma, lower=config.positive_minimum)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1,X2 = self._check_input(X1,X2)

        sqdist = self.squared_distance(X1,X2)  # NxMxD
        power = 1.0+0.5*torch.tensordot(sqdist, 1.0/self.l()**2, dims=1)/self.alpha  # NxM
        return self.sigma()**2 * torch.pow(power,-self.alpha)

class PeriodicKernel(Kernel):
    def __init__(self, input_dims, active_dims=None, name="Periodic"):
        super(PeriodicKernel, self).__init__(input_dims, active_dims, name)

        l = torch.rand(input_dims)
        p = torch.rand(1)
        sigma = torch.rand(1)

        self.l = Parameter(l, lower=config.positive_minimum)
        self.p = Parameter(p, lower=config.positive_minimum)
        self.sigma = Parameter(sigma, lower=config.positive_minimum)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1,X2 = self._check_input(X1,X2)

        sin = torch.sin(np.pi * self.distance(X1,X2) / self.p())  # NxMxD
        exp = torch.exp(-2.0 * torch.tensordot(sin**2, self.l()**2, dims=1))  # NxM
        return self.sigma()**2 * exp

class SpectralKernel(Kernel):
    def __init__(self, input_dims, active_dims=None, name="SM"):
        super(SpectralKernel, self).__init__(input_dims, active_dims, name)

        weight = torch.rand(1)
        mean = torch.rand(input_dims)
        variance = torch.ones(input_dims)

        self.weight = Parameter(weight, lower=config.positive_minimum)
        self.mean = Parameter(mean, lower=config.positive_minimum)
        self.variance = Parameter(variance, lower=config.positive_minimum)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1,X2 = self._check_input(X1,X2)
    
        tau = self.distance(X1,X2)  # NxMxD
        exp = torch.exp(-2.0*np.pi**2 * tau**2 * self.variance().reshape(1,1,-1))  # NxMxD
        cos = torch.cos(2.0*np.pi * tau * self.mean().reshape(1,1,-1))  # NxMxD
        return self.weight() * torch.prod(exp * cos, dim=2)

class MaternKernel(Kernel):
    def __init__(self, nu=0.5, input_dims=None, active_dims=None, name="Matérn"):
        super(MaternKernel, self).__init__(input_dims, active_dims, name)

        if nu not in [0.5, 1.5, 2.5]:
            raise ValueError("nu parameter must be 0.5, 1.5, or 2.5")

        l = torch.rand(input_dims)
        sigma = torch.rand(1)

        self.nu = nu
        self.l = Parameter(l, lower=1e-6)
        self.sigma = Parameter(sigma, lower=1e-6)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1,X2 = self._check_input(X1,X2)
        if X2 is None:
            X2 = X1

        dist = torch.abs(torch.tensordot(self.distance(X1,X2), 1.0/self.l(), dims=1))
        if self.nu == 0.5:
            constant = 1.0
        elif self.nu == 1.5:
            constant = 1.0 + np.sqrt(3.0)*dist
        elif self.nu == 2.5:
            constant = 1.0 + np.sqrt(5.0)*dist + 5.0/3.0*dist**2
        return self.sigma()**2 * constant * torch.exp(-np.sqrt(self.nu*2.0)*dist)
