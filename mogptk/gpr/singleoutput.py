import torch
import numpy as np
from . import Kernel, Parameter, config

class ConstantKernel(Kernel):
    """
    A constant or bias kernel given by

    $$ K(x,x') = \\sigma^2 $$

    with \\(\\sigma^2\\) the magnitude.

    Args:
        input_dims (int): Number of input dimensions.
        active_dims (list of int): Indices of active dimensions of shape (input_dims,).
        name (str): Kernel name.
    """
    def __init__(self, input_dims=1, active_dims=None, name="Constant"):
        super().__init__(input_dims, active_dims, name)

        self.magnitude = Parameter(1.0, lower=config.positive_minimum)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1, X2 = self._active_input(X1, X2)
        if X2 is None:
            X2 = X1
        return self.magnitude()**2 * torch.ones(X1.shape[0], X2.shape[0], dtype=config.dtype, device=config.device)

    def K_diag(self, X1):
        # X has shape (data_points,input_dims)
        X1, _ = self._active_input(X1)
        return self.magnitude()**2 * torch.ones(X1.shape[0], dtype=config.dtype, device=config.device)

class WhiteKernel(Kernel):
    """
    A white kernel given by

    $$ K(x,x') = \\sigma^2 I $$

    with \\(\\sigma^2\\) the magnitude.

    Args:
        input_dims (int): Number of input dimensions.
        active_dims (list of int): Indices of active dimensions of shape (input_dims,).
        name (str): Kernel name.
    """
    def __init__(self, input_dims=1, active_dims=None, name="White"):
        super().__init__(input_dims, active_dims, name)

        self.magnitude = Parameter(1.0, lower=config.positive_minimum)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1, X2 = self._active_input(X1, X2)
        if X2 is None:
            return self.magnitude()**2 * torch.eye(X1.shape[0], X1.shape[0], dtype=config.dtype, device=config.device)
        return torch.zeros(X1.shape[0], X2.shape[0], dtype=config.dtype, device=config.device)

    def K_diag(self, X1):
        # X has shape (data_points,input_dims)
        X1, _ = self._active_input(X1)
        return self.magnitude()**2 * torch.ones(X1.shape[0], dtype=config.dtype, device=config.device)

class LinearKernel(Kernel):
    """
    A linear kernel given by

    $$ K(x,x') = \\sigma^2 xx'^T + c $$

    with \\(\\sigma^2\\) the magnitude and \\(c\\) the bias.

    Args:
        input_dims (int): Number of input dimensions.
        active_dims (list of int): Indices of active dimensions of shape (input_dims,).
        name (str): Kernel name.
    """
    def __init__(self, input_dims=1, active_dims=None, name="Linear"):
        super().__init__(input_dims, active_dims, name)

        self.constant = Parameter(0.0, lower=0.0)
        self.magnitude = Parameter(1.0, lower=config.positive_minimum)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1, X2 = self._active_input(X1, X2)
        if X2 is None:
            X2 = X1
        return self.magnitude()**2 * X1.mm(X2.T) + self.constant()

    def K_diag(self, X1):
        # X has shape (data_points,input_dims)
        X1, _ = self._active_input(X1)
        return self.magnitude()**2 * X1.square().sum(dim=1) + self.constant()

class PolynomialKernel(Kernel):
    """
    A polynomial kernel given by

    $$ K(x,x') = (\\sigma^2 xx'^T + c)^d $$

    with \\(\\sigma^2\\) the magnitude, \\(c\\) the bias, and \\(d\\) the degree of the polynomial. When \\(d\\) is 1 this becomes equivalent to the linear kernel.

    Args:
        degree (int): Degree of the polynomial.
        input_dims (int): Number of input dimensions.
        active_dims (list of int): Indices of active dimensions of shape (input_dims,).
        name (str): Kernel name.
    """
    def __init__(self, degree, input_dims=1, active_dims=None, name="Polynomial"):
        super().__init__(input_dims, active_dims, name)

        self.degree = degree
        self.magnitude = Parameter(1.0, lower=config.positive_minimum)
        self.offset = Parameter(0.0, lower=0.0)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1, X2 = self._active_input(X1, X2)
        if X2 is None:
            X2 = X1
        return (self.magnitude()**2 * X1.mm(X2.T) + self.offset())**self.degree

    def K_diag(self, X1):
        # X has shape (data_points,input_dims)
        X1, _ = self._active_input(X1)
        return (self.magnitude()**2 * X1.square().sum(dim=1) + self.offset())**self.degree

class PhiKernel(Kernel):
    """
    A kernel determined by a function \\(\\phi\\) given by

    $$ K(x,x') = \\phi(x) \\sigma^2 \\phi(x') $$

    with \\(\\sigma^2\\) the magnitude and \\(\\phi\\) the function.

    Args:
        phi (function): Function that takes (data_points,input_dims) as input and (data_points,feature_dims) as output. Input and output must be tensors.
        input_dims (int): Number of input dimensions.
        active_dims (list of int): Indices of active dimensions of shape (input_dims,).
        name (str): Kernel name.
    """
    def __init__(self, phi, input_dims=1, active_dims=None, name="Phi"):
        super().__init__(input_dims, active_dims, name)

        out = phi(torch.ones(42, input_dims, dtype=config.dtype, device=config.device))
        if not torch.is_tensor(out) or out.dtype != config.dtype or out.device != config.device:
            raise ValueError("phi must return a tensor of the same dtype and device as the input")
        if out.ndim != 2 or out.shape[0] != 42:
            raise ValueError("phi must take (data_points,input_dims) as input, and return (data_points,feature_dims) as output")

        feature_dims = out.shape[1]
        magnitude = torch.ones(feature_dims)

        self.magnitude = Parameter(magnitude, lower=config.positive_minimum)
        self.phi = phi

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1, X2 = self._active_input(X1, X2)
        variance = (self.magnitude()**2).diagflat()
        if X2 is None:
            X1 = self.phi(X1)
            return X1.mm(variance.mm(X1.T))
        else:
            return self.phi(X1).mm(variance.mm(self.phi(X2).T))

class ExponentialKernel(Kernel):
    """
    An exponential kernel given by

    $$ K(x,x') = \\sigma^2 e^{-\\frac{|x-x'|}{2l}} $$

    with \\(\\sigma^2\\) the magnitude, \\(l\\) the lengthscale.

    Args:
        input_dims (int): Number of input dimensions.
        active_dims (list of int): Indices of active dimensions of shape (input_dims,).
        name (str): Kernel name.
    """
    def __init__(self, input_dims=1, active_dims=None, name="Exponential"):
        super().__init__(input_dims, active_dims, name)

        magnitude = 1.0
        l = torch.ones(input_dims)

        self.magnitude = Parameter(magnitude, lower=config.positive_minimum)
        self.l = Parameter(l, lower=config.positive_minimum)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1, X2 = self._active_input(X1, X2)
        dist = torch.abs(self.distance(X1,X2))  # NxMxD
        exp = -0.5*torch.tensordot(dist, 1.0/self.l(), dims=1)  # NxM
        return self.magnitude()**2 * torch.exp(exp)

    def K_diag(self, X1):
        # X has shape (data_points,input_dims)
        X1, _ = self._active_input(X1)
        return self.magnitude()**2 * torch.ones(X1.shape[0], dtype=config.dtype, device=config.device)

class SquaredExponentialKernel(Kernel):
    """
    A squared exponential kernel given by

    $$ K(x,x') = \\sigma^2 e^{-\\frac{|x-x'|^2}{2l^2}} $$

    with \\(\\sigma^2\\) the magnitude and \\(l\\) the lengthscale.

    Args:
        input_dims (int): Number of input dimensions.
        active_dims (list of int): Indices of active dimensions of shape (input_dims,).
        name (str): Kernel name.
    """
    def __init__(self, input_dims=1, active_dims=None, name="SE"):
        super().__init__(input_dims, active_dims, name)

        magnitude = 1.0
        l = torch.ones(input_dims)

        self.magnitude = Parameter(magnitude, lower=config.positive_minimum)
        self.l = Parameter(l, lower=config.positive_minimum)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1, X2 = self._active_input(X1, X2)
        sqdist = self.squared_distance(X1,X2)  # NxMxD
        exp = -0.5*torch.tensordot(sqdist, 1.0/self.l()**2, dims=1)  # NxM
        return self.magnitude()**2 * torch.exp(exp)

    def K_diag(self, X1):
        # X has shape (data_points,input_dims)
        X1, _ = self._active_input(X1)
        return self.magnitude()**2 * torch.ones(X1.shape[0], dtype=config.dtype, device=config.device)

class RationalQuadraticKernel(Kernel):
    """
    A rational quadratic kernel given by

    $$ K(x,x') = \\sigma^2 \\left(1 + \\frac{|x-x'|^2}{2l^2}\\right)^{-\\alpha} $$

    with \\(\\sigma^2\\) the magnitude, \\(l\\) the lengthscale, and \\(\\alpha\\) the relative weighting of small-scale and large-scale fluctuations. When \\(\\alpha \\to \\infty\\) this kernel becomes equivalent to the squared exponential kernel.

    Args:
        input_dims (int): Number of input dimensions.
        active_dims (list of int): Indices of active dimensions of shape (input_dims,).
        name (str): Kernel name.
    """
    def __init__(self, alpha=1.0, input_dims=1, active_dims=None, name="RQ"):
        super().__init__(input_dims, active_dims, name)

        magnitude = 1.0
        l = torch.ones(input_dims)

        self.alpha = alpha
        self.magnitude = Parameter(magnitude, lower=config.positive_minimum)
        self.l = Parameter(l, lower=config.positive_minimum)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1, X2 = self._active_input(X1, X2)
        sqdist = self.squared_distance(X1,X2)  # NxMxD
        power = 1.0+0.5*torch.tensordot(sqdist, 1.0/self.l()**2, dims=1)/self.alpha  # NxM
        return self.magnitude()**2 * torch.pow(power,-self.alpha)

    def K_diag(self, X1):
        # X has shape (data_points,input_dims)
        X1, _ = self._active_input(X1)
        return self.magnitude()**2 * torch.ones(X1.shape[0], dtype=config.dtype, device=config.device)

class PeriodicKernel(Kernel):
    """
    A periodic kernel given by

    $$ K(x,x') = \\sigma^2 e^{-\\frac{2\\sin^2(\\pi |x-x'| / p)}{l^2}} $$

    with \\(\\sigma^2\\) the magnitude, \\(l\\) the lengthscale, and \\(p\\) the period parameter.

    Args:
        input_dims (int): Number of input dimensions.
        active_dims (list of int): Indices of active dimensions of shape (input_dims,).
        name (str): Kernel name.
    """
    def __init__(self, input_dims=1, active_dims=None, name="Periodic"):
        super().__init__(input_dims, active_dims, name)

        magnitude = 1.0
        l = torch.ones(input_dims)
        p = 1.0

        self.magnitude = Parameter(magnitude, lower=config.positive_minimum)
        self.l = Parameter(l, lower=config.positive_minimum)
        self.p = Parameter(p, lower=config.positive_minimum)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1, X2 = self._active_input(X1, X2)
        tau = self.distance(X1,X2)
        sin = torch.sin(np.pi * tau / self.p())  # NxMxD
        exp = -2.0 * torch.tensordot(sin**2, 1.0/self.l()**2, dims=1)  # NxM
        return self.magnitude()**2 * torch.exp(exp)

    def K_diag(self, X1):
        # X has shape (data_points,input_dims)
        X1, _ = self._active_input(X1)
        return self.magnitude()**2 * torch.ones(X1.shape[0], dtype=config.dtype, device=config.device)

class LocallyPeriodicKernel(Kernel):
    """
    A locally periodic kernel given by

    $$ K(x,x') = \\sigma^2 e^{-\\frac{2\\sin^2(\\pi |x-x'| / p)}{l^2}} e^{-\\frac{|x-x'|^2}{2l^2}} $$

    with \\(\\sigma^2\\) the magnitude, \\(l\\) the lengthscale, and \\(p\\) the period parameter.

    Args:
        input_dims (int): Number of input dimensions.
        active_dims (list of int): Indices of active dimensions of shape (input_dims,).
        name (str): Kernel name.
    """
    def __init__(self, input_dims=1, active_dims=None, name="LocallyPeriodic"):
        super().__init__(input_dims, active_dims, name)

        magnitude = 1.0
        l = torch.ones(input_dims)
        p = 1.0

        self.magnitude = Parameter(magnitude, lower=config.positive_minimum)
        self.l = Parameter(l, lower=config.positive_minimum)
        self.p = Parameter(p, lower=config.positive_minimum)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1, X2 = self._active_input(X1, X2)
        tau = self.distance(X1,X2)
        sqdist = self.squared_distance(X1,X2)
        sin = torch.sin(np.pi * tau / self.p())  # NxMxD
        exp1 = -2.0 * torch.tensordot(sin**2, 1.0/self.l()**2, dims=1)  # NxM
        exp2 = -0.5 * torch.tensordot(sqdist, 1.0/self.l()**2, dims=1)  # NxM
        return self.magnitude()**2 * torch.exp(exp1) * torch.exp(exp2)

    def K_diag(self, X1):
        # X has shape (data_points,input_dims)
        X1, _ = self._active_input(X1)
        return self.magnitude()**2 * torch.ones(X1.shape[0], dtype=config.dtype, device=config.device)

class CosineKernel(Kernel):
    """
    A cosine periodic kernel given by

    $$ K(x,x') = \\sigma^2 \\cos(2\\pi |x-x'| / l) $$

    with \\(\\sigma^2\\) the magnitude and \\(l\\) the lengthscale.

    Args:
        input_dims (int): Number of input dimensions.
        active_dims (list of int): Indices of active dimensions of shape (input_dims,).
        name (str): Kernel name.
    """
    def __init__(self, input_dims=1, active_dims=None, name="Cosine"):
        super().__init__(input_dims, active_dims, name)

        magnitude = 1.0
        l = torch.ones(input_dims)

        self.magnitude = Parameter(magnitude, lower=config.positive_minimum)
        self.l = Parameter(l, lower=config.positive_minimum)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1, X2 = self._active_input(X1, X2)
        tau = self.distance(X1,X2)
        cos = 2.0*np.pi * torch.tensordot(tau, 1.0/self.l(), dims=1)  # NxM
        return self.magnitude()**2 * torch.cos(cos)

    def K_diag(self, X1):
        # X has shape (data_points,input_dims)
        X1, _ = self._active_input(X1)
        return self.magnitude()**2 * torch.ones(X1.shape[0], dtype=config.dtype, device=config.device)

class SincKernel(Kernel):
    """
    A sinc kernel given by

    $$ K(x,x') = \\sigma^2 \\frac{\\sin(\\Delta |x-x'|)}{\\Delta |x-x'|} \\cos(2\\pi \\xi_0 |x-x'|) $$

    with \\(\\sigma^2\\) the magnitude, \\(\\Delta\\) the bandwidth, and \\(\\xi_0\\) the frequency.

    Args:
        input_dims (int): Number of input dimensions.
        active_dims (list of int): Indices of active dimensions of shape (input_dims,).
        name (str): Kernel name.
    """
    def __init__(self, input_dims=1, active_dims=None, name="Sinc"):
        super().__init__(input_dims, active_dims, name)

        magnitude = 1.0
        frequency = torch.ones(input_dims)
        bandwidth = torch.ones(input_dims, dtype=config.dtype, device=config.device)

        self.magnitude = Parameter(magnitude, lower=config.positive_minimum)
        self.frequency = Parameter(frequency, lower=config.positive_minimum)
        self.bandwidth = Parameter(bandwidth, lower=config.positive_minimum)

    def _sinc(self, x):
        x = torch.where(x == 0.0, 1e-20 * torch.ones_like(x), x)
        return torch.sin(np.pi*x) / (np.pi*x)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1, X2 = self._active_input(X1, X2)
        tau = self.distance(X1,X2)  # NxMxD
        sinc = torch.tensordot(tau, self.bandwidth(), dims=1)  # NxM
        cos = 2.0*np.pi * torch.tensordot(tau, self.frequency(), dims=1)  # NxM
        return self.magnitude()**2 * self._sinc(sinc) * torch.cos(cos)

    def K_diag(self, X1):
        # X has shape (data_points,input_dims)
        X1, _ = self._active_input(X1)
        return self.magnitude()**2 * torch.ones(X1.shape[0], dtype=config.dtype, device=config.device)

class SpectralKernel(Kernel):
    """
    A spectral kernel given by

    $$ K(x,x') = \\sigma^2 e^{-2\\pi^2 \\Sigma |x-x'|^2} \\cos(2\\pi \\mu |x-x'|) $$

    with \\(\\sigma^2\\) the magnitude, \\(\\Sigma\\) the variance, and \\(\\mu\\) the mean.

    Args:
        input_dims (int): Number of input dimensions.
        active_dims (list of int): Indices of active dimensions of shape (input_dims,).
        name (str): Kernel name.
    """
    def __init__(self, input_dims=1, active_dims=None, name="SM"):
        super().__init__(input_dims, active_dims, name)

        magnitude = 1.0
        mean = torch.zeros(input_dims)
        variance = torch.ones(input_dims)

        self.magnitude = Parameter(magnitude, lower=config.positive_minimum)
        self.mean = Parameter(mean, lower=config.positive_minimum)
        self.variance = Parameter(variance, lower=config.positive_minimum)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1, X2 = self._active_input(X1, X2)
        tau = self.distance(X1,X2)  # NxMxD
        exp = -2.0*np.pi**2 * tau**2 * self.variance().reshape(1,1,-1)  # NxMxD
        cos = 2.0*np.pi * tau * self.mean().reshape(1,1,-1)  # NxMxD
        return self.magnitude()**2 * torch.prod(torch.exp(exp) * torch.cos(cos), dim=2)

    def K_diag(self, X1):
        # X has shape (data_points,input_dims)
        X1, _ = self._active_input(X1)
        return self.magnitude()**2 * torch.ones(X1.shape[0], dtype=config.dtype, device=config.device)

class MaternKernel(Kernel):
    """
    A Matérn kernel given by

    $$ K(x,x') = \\sigma^2 c e^{-\\sqrt{2\\nu |x-x'| / l}} $$

    with \\(\\sigma^2\\) the magnitude, \\(l\\) the lengthscale, and \\(c\\) depending on \\(\\nu\\) is either \\(1.0\\) for \\(\\nu = 0.5\\), or \\(1.0 + \\sqrt{3}|x-x'|/l\\) for \\(\\nu = 1.5\\), or \\(1.0 + \\sqrt{5}|x-x'|/l + \\frac{5|x-x'|^2}{3l^2}\\).

    Args:
        nu (float): Parameter that must be 0.5, 1.5, or 2.5.
        input_dims (int): Number of input dimensions.
        active_dims (list of int): Indices of active dimensions of shape (input_dims,).
        name (str): Kernel name.
    """
    def __init__(self, nu=0.5, input_dims=1, active_dims=None, name="Matérn"):
        super().__init__(input_dims, active_dims, name)

        if nu not in [0.5, 1.5, 2.5]:
            raise ValueError("nu parameter must be 0.5, 1.5, or 2.5")

        magnitude = 1.0
        l = torch.ones(input_dims)

        self.nu = nu
        self.magnitude = Parameter(magnitude, lower=1e-6)
        self.l = Parameter(l, lower=1e-6)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1, X2 = self._active_input(X1, X2)
        if X2 is None:
            X2 = X1

        dist = torch.abs(torch.tensordot(self.distance(X1,X2), 1.0/self.l(), dims=1))
        if self.nu == 0.5:
            constant = 1.0
        elif self.nu == 1.5:
            constant = 1.0 + np.sqrt(3.0)*dist
        elif self.nu == 2.5:
            constant = 1.0 + np.sqrt(5.0)*dist + 5.0/3.0*dist**2
        return self.magnitude()**2 * constant * torch.exp(-np.sqrt(self.nu*2.0) * dist)

    def K_diag(self, X1):
        # X has shape (data_points,input_dims)
        X1, _ = self._active_input(X1)
        return self.magnitude()**2 * torch.ones(X1.shape[0], dtype=config.dtype, device=config.device)
