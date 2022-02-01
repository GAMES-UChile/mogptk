import torch
import numpy as np
from . import Kernel, Parameter, config

class ConstantKernel(Kernel):
    def __init__(self, input_dims=1, active_dims=None, name="Constant"):
        super().__init__(input_dims, active_dims, name)

        sigma = torch.rand(1)

        self.sigma = Parameter(sigma, lower=config.positive_minimum)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1, X2 = self._active_input(X1, X2)
        if X2 is None:
            X2 = X1
        return self.sigma()**2 * torch.ones(X1.shape[0], X2.shape[0], dtype=config.dtype, device=config.device)

    def K_diag(self, X1):
        # X has shape (data_points,input_dims)
        X1, _ = self._active_input(X1)
        return self.sigma()**2 * torch.ones(X1.shape[0], dtype=config.dtype, device=config.device)

class WhiteKernel(Kernel):
    def __init__(self, input_dims=1, active_dims=None, name="White"):
        super().__init__(input_dims, active_dims, name)

        sigma = torch.rand(1)

        self.sigma = Parameter(sigma, lower=config.positive_minimum)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1, X2 = self._active_input(X1, X2)
        if X2 is None:
            return self.sigma()**2 * torch.eye(X1.shape[0], X1.shape[0], dtype=config.dtype, device=config.device)
        return torch.zeros(X1.shape[0], X2.shape[0], dtype=config.dtype, device=config.device)

    def K_diag(self, X1):
        # X has shape (data_points,input_dims)
        X1, _ = self._active_input(X1)
        return self.sigma()**2 * torch.ones(X1.shape[0], dtype=config.dtype, device=config.device)

class LinearKernel(Kernel):
    def __init__(self, input_dims=1, active_dims=None, name="Linear"):
        super().__init__(input_dims, active_dims, name)

        constant = torch.rand(1)
        sigma = torch.rand(1)

        self.constant = Parameter(constant, lower=0.0)
        self.sigma = Parameter(sigma, lower=config.positive_minimum)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1, X2 = self._active_input(X1, X2)
        if X2 is None:
            X2 = X1
        return self.sigma()**2 * X1.mm(X2.T) + self.constant()

    def K_diag(self, X1):
        # X has shape (data_points,input_dims)
        X1, _ = self._active_input(X1)
        return self.sigma()**2 * X1.square().sum(dim=1) + self.constant()

class PolynomialKernel(Kernel):
    def __init__(self, degree, input_dims=1, active_dims=None, name="Polynomial"):
        super().__init__(input_dims, active_dims, name)

        offset = torch.rand(1)
        sigma = torch.rand(1)

        self.degree = degree
        self.offset = Parameter(offset, lower=0.0)
        self.sigma = Parameter(sigma, lower=config.positive_minimum)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1, X2 = self._active_input(X1, X2)
        if X2 is None:
            X2 = X1
        return (self.sigma()**2 * X1.mm(X2.T) + self.offset())**self.degree

    def K_diag(self, X1):
        # X has shape (data_points,input_dims)
        X1, _ = self._active_input(X1)
        return (self.sigma()**2 * X1.square().sum(dim=1) + self.offset())**self.degree

class PhiKernel(Kernel):
    def __init__(self, phi, input_dims=1, active_dims=None, name="Phi"):
        super().__init__(input_dims, active_dims, name)

        out = phi(torch.ones(42, input_dims, dtype=config.dtype, device=config.device))
        if not torch.is_tensor(out) or out.dtype != config.dtype or out.device != config.device:
            raise ValueError("phi must return a tensor of the same dtype and device as the input")
        if len(out.shape) != 2 or out.shape[0] != 42:
            raise ValueError("phi must take (data_points,input_dims) as input, and return (data_points,feature_dims) as output")

        feature_dims = out.shape[1]
        sigma = torch.ones(feature_dims)

        self.phi = phi
        self.sigma = Parameter(sigma, lower=config.positive_minimum)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1, X2 = self._active_input(X1, X2)
        variance = self.sigma()**2
        if X2 is None:
            X1 = self.phi(X1)
            return X1.mm(variance.diagflat().mm(X1.T))
        else:
            return self.phi(X1).mm(variance.diagflat().mm(self.phi(X2).T))

    # TODO: K_diag

class ExponentialKernel(Kernel):
    def __init__(self, input_dims=1, active_dims=None, name="Exponential"):
        super().__init__(input_dims, active_dims, name)

        l = torch.rand(input_dims)
        sigma = torch.rand(1)

        self.l = Parameter(l, lower=config.positive_minimum)
        self.sigma = Parameter(sigma, lower=config.positive_minimum)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1, X2 = self._active_input(X1, X2)
        dist = torch.abs(self.distance(X1,X2))  # NxMxD
        exp = -0.5*torch.tensordot(dist, 1.0/self.l(), dims=1)  # NxM
        return self.sigma()**2 * torch.exp(exp)

    def K_diag(self, X1):
        # X has shape (data_points,input_dims)
        X1, _ = self._active_input(X1)
        return self.sigma()**2 * torch.ones(X1.shape[0], dtype=config.dtype, device=config.device)

class SquaredExponentialKernel(Kernel):
    def __init__(self, input_dims=1, active_dims=None, name="SE"):
        super().__init__(input_dims, active_dims, name)

        l = torch.rand(input_dims)
        sigma = torch.rand(1)

        self.l = Parameter(l, lower=config.positive_minimum)
        self.sigma = Parameter(sigma, lower=config.positive_minimum)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1, X2 = self._active_input(X1, X2)
        sqdist = self.squared_distance(X1,X2)  # NxMxD
        exp = -0.5*torch.tensordot(sqdist, 1.0/self.l()**2, dims=1)  # NxM
        return self.sigma()**2 * torch.exp(exp)

    def K_diag(self, X1):
        # X has shape (data_points,input_dims)
        X1, _ = self._active_input(X1)
        return self.sigma()**2 * torch.ones(X1.shape[0], dtype=config.dtype, device=config.device)

class RationalQuadraticKernel(Kernel):
    def __init__(self, alpha=1.0, input_dims=1, active_dims=None, name="RQ"):
        super().__init__(input_dims, active_dims, name)

        l = torch.rand(input_dims)
        sigma = torch.rand(1)

        self.alpha = alpha
        self.l = Parameter(l, lower=config.positive_minimum)
        self.sigma = Parameter(sigma, lower=config.positive_minimum)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1, X2 = self._active_input(X1, X2)
        sqdist = self.squared_distance(X1,X2)  # NxMxD
        power = 1.0+0.5*torch.tensordot(sqdist, 1.0/self.l()**2, dims=1)/self.alpha  # NxM
        return self.sigma()**2 * torch.pow(power,-self.alpha)

    def K_diag(self, X1):
        # X has shape (data_points,input_dims)
        X1, _ = self._active_input(X1)
        return self.sigma()**2 * torch.ones(X1.shape[0], dtype=config.dtype, device=config.device)

class PeriodicKernel(Kernel):
    def __init__(self, input_dims=1, active_dims=None, name="Periodic"):
        super().__init__(input_dims, active_dims, name)
        # TODO: make nested by SE

        l = torch.rand(input_dims)
        p = torch.rand(1)
        sigma = torch.rand(1)

        self.l = Parameter(l, lower=config.positive_minimum)
        self.p = Parameter(p, lower=config.positive_minimum)
        self.sigma = Parameter(sigma, lower=config.positive_minimum)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1, X2 = self._active_input(X1, X2)
        tau = self.distance(X1,X2)
        sin = torch.sin(np.pi * tau / self.p())  # NxMxD
        exp = -2.0 * torch.tensordot(sin**2, 1.0/self.l()**2, dims=1)  # NxM
        return self.sigma()**2 * torch.exp(exp)

    def K_diag(self, X1):
        # X has shape (data_points,input_dims)
        X1, _ = self._active_input(X1)
        return self.sigma()**2 * torch.ones(X1.shape[0], dtype=config.dtype, device=config.device)

class LocallyPeriodicKernel(Kernel):
    def __init__(self, input_dims=1, active_dims=None, name="LocallyPeriodic"):
        super().__init__(input_dims, active_dims, name)

        l = torch.rand(input_dims)
        p = torch.rand(1)
        sigma = torch.rand(1)

        self.l = Parameter(l, lower=config.positive_minimum)
        self.p = Parameter(p, lower=config.positive_minimum)
        self.sigma = Parameter(sigma, lower=config.positive_minimum)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1, X2 = self._active_input(X1, X2)
        tau = self.distance(X1,X2)
        sqdist = self.squared_distance(X1,X2)
        sin = torch.sin(np.pi * tau / self.p())  # NxMxD
        exp1 = -2.0 * torch.tensordot(sin**2, 1.0/self.l()**2, dims=1)  # NxM
        exp2 = -0.5 * torch.tensordot(sqdist, 1.0/self.l()**2, dims=1)  # NxM
        return self.sigma()**2 * torch.exp(exp1) * torch.exp(exp2)

    def K_diag(self, X1):
        # X has shape (data_points,input_dims)
        X1, _ = self._active_input(X1)
        return self.sigma()**2 * torch.ones(X1.shape[0], dtype=config.dtype, device=config.device)

class CosineKernel(Kernel):
    def __init__(self, input_dims=1, active_dims=None, name="Cosine"):
        super().__init__(input_dims, active_dims, name)

        l = torch.rand(input_dims)
        sigma = torch.rand(1)

        self.l = Parameter(l, lower=config.positive_minimum)
        self.sigma = Parameter(sigma, lower=config.positive_minimum)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1, X2 = self._active_input(X1, X2)
        tau = self.distance(X1,X2)
        cos = 2.0*np.pi * torch.tensordot(tau, 1.0/self.l(), dims=1)  # NxM
        return self.sigma()**2 * torch.cos(cos)

    def K_diag(self, X1):
        # X has shape (data_points,input_dims)
        X1, _ = self._active_input(X1)
        return self.sigma()**2 * torch.ones(X1.shape[0], dtype=config.dtype, device=config.device)

class SincKernel(Kernel):
    def __init__(self, bandwidth=1.0, input_dims=1, active_dims=None, name="Sinc"):
        super().__init__(input_dims, active_dims, name)

        frequency = torch.rand(input_dims)
        bandwidth = bandwidth * torch.ones(input_dims, dtype=config.dtype, device=config.device)
        sigma = torch.rand(1)

        self.frequency = Parameter(frequency)
        self.bandwidth = Parameter(bandwidth)
        self.sigma = Parameter(sigma)

    def _sinc(self, x):
        x = torch.where(x == 0.0, 1e-20 * torch.ones_like(x), x)
        return torch.sin(np.pi*x) / (np.pi*x)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1, X2 = self._active_input(X1, X2)
        tau = self.distance(X1,X2)  # NxMxD
        sinc = torch.tensordot(tau, self.bandwidth(), dims=1)  # NxM
        cos = 2.0*np.pi * torch.tensordot(tau, self.frequency(), dims=1)  # NxM
        return self.sigma()**2 * self._sinc(sinc) * torch.cos(cos)

    def K_diag(self, X1):
        # X has shape (data_points,input_dims)
        X1, _ = self._active_input(X1)
        return self.sigma()**2 * torch.ones(X1.shape[0], dtype=config.dtype, device=config.device)

class SpectralKernel(Kernel):
    def __init__(self, input_dims=1, active_dims=None, name="SM"):
        super().__init__(input_dims, active_dims, name)

        weight = torch.rand(1)
        mean = torch.rand(input_dims)
        variance = torch.ones(input_dims)

        self.weight = Parameter(weight, lower=config.positive_minimum)
        self.mean = Parameter(mean, lower=config.positive_minimum)
        self.variance = Parameter(variance, lower=config.positive_minimum)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1, X2 = self._active_input(X1, X2)
        tau = self.distance(X1,X2)  # NxMxD
        exp = -2.0*np.pi**2 * tau**2 * self.variance().reshape(1,1,-1)  # NxMxD
        cos = 2.0*np.pi * tau * self.mean().reshape(1,1,-1)  # NxMxD
        return self.weight() * torch.prod(torch.exp(exp) * torch.cos(cos), dim=2)

    def K_diag(self, X1):
        # X has shape (data_points,input_dims)
        X1, _ = self._active_input(X1)
        return self.weight() * torch.ones(X1.shape[0], dtype=config.dtype, device=config.device)

class MaternKernel(Kernel):
    def __init__(self, nu=0.5, input_dims=1, active_dims=None, name="Matérn"):
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
        X1, X2 = self._active_input(X1, X2)
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

    def K_diag(self, X1):
        # X has shape (data_points,input_dims)
        X1, _ = self._active_input(X1)
        return self.sigma()**2 * torch.ones(X1.shape[0], dtype=config.dtype, device=config.device)
