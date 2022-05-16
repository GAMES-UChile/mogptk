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

    Attributes:
        magnitude (mogptk.gpr.parameter.Parameter): Magnitude \\(\\sigma^2\\) a scalar.
    """
    def __init__(self, input_dims=1, active_dims=None, name="Constant"):
        super().__init__(input_dims, active_dims, name)

        self.magnitude = Parameter(1.0, lower=config.positive_minimum)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1, X2 = self._active_input(X1, X2)
        if X2 is None:
            X2 = X1
        return self.magnitude() * torch.ones(X1.shape[0], X2.shape[0], dtype=config.dtype, device=config.device)

    def K_diag(self, X1):
        # X has shape (data_points,input_dims)
        X1, _ = self._active_input(X1)
        return self.magnitude().repeat(X1.shape[0])

class WhiteKernel(Kernel):
    """
    A white kernel given by

    $$ K(x,x') = \\sigma^2 I $$

    with \\(\\sigma^2\\) the magnitude.

    Args:
        input_dims (int): Number of input dimensions.
        active_dims (list of int): Indices of active dimensions of shape (input_dims,).
        name (str): Kernel name.

    Attributes:
        magnitude (mogptk.gpr.parameter.Parameter): Magnitude \\(\\sigma^2\\) a scalar.
    """
    def __init__(self, input_dims=1, active_dims=None, name="White"):
        super().__init__(input_dims, active_dims, name)

        self.magnitude = Parameter(1.0, lower=config.positive_minimum)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1, X2 = self._active_input(X1, X2)
        if X2 is None:
            return self.magnitude() * torch.eye(X1.shape[0], X1.shape[0], dtype=config.dtype, device=config.device)
        return torch.zeros(X1.shape[0], X2.shape[0], dtype=config.dtype, device=config.device)

    def K_diag(self, X1):
        # X has shape (data_points,input_dims)
        X1, _ = self._active_input(X1)
        return self.magnitude().repeat(X1.shape[0])

class LinearKernel(Kernel):
    """
    A linear kernel given by

    $$ K(x,x') = \\sigma^2 xx'^T + c $$

    with \\(\\sigma^2\\) the magnitude and \\(c\\) the bias.

    Args:
        input_dims (int): Number of input dimensions.
        active_dims (list of int): Indices of active dimensions of shape (input_dims,).
        name (str): Kernel name.

    Attributes:
        bias (mogptk.gpr.parameter.Parameter): Bias \\(c\\) a scalar.
        magnitude (mogptk.gpr.parameter.Parameter): Magnitude \\(\\sigma^2\\) a scalar.
    """
    def __init__(self, input_dims=1, active_dims=None, name="Linear"):
        super().__init__(input_dims, active_dims, name)

        self.bias = Parameter(0.0, lower=0.0)
        self.magnitude = Parameter(1.0, lower=config.positive_minimum)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1, X2 = self._active_input(X1, X2)
        if X2 is None:
            X2 = X1
        return self.magnitude() * X1.mm(X2.T) + self.bias()

    def K_diag(self, X1):
        # X has shape (data_points,input_dims)
        X1, _ = self._active_input(X1)
        return self.magnitude() * X1.square().sum(dim=1) + self.bias()

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

    Attributes:
        degree (float): Degree \\(d\\).
        bias (mogptk.gpr.parameter.Parameter): Bias \\(c\\) a scalar.
        magnitude (mogptk.gpr.parameter.Parameter): Magnitude \\(\\sigma^2\\) a scalar.
    """
    def __init__(self, degree, input_dims=1, active_dims=None, name="Polynomial"):
        super().__init__(input_dims, active_dims, name)

        self.degree = degree
        self.bias = Parameter(0.0, lower=0.0)
        self.magnitude = Parameter(1.0, lower=config.positive_minimum)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1, X2 = self._active_input(X1, X2)
        if X2 is None:
            X2 = X1
        return (self.magnitude() * X1.mm(X2.T) + self.bias())**self.degree

    def K_diag(self, X1):
        # X has shape (data_points,input_dims)
        X1, _ = self._active_input(X1)
        return (self.magnitude() * X1.square().sum(dim=1) + self.bias())**self.degree

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

    Attributes:
        magnitude (mogptk.gpr.parameter.Parameter): Magnitude \\(\\sigma^2\\) a scalar.
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
        variance = self.magnitude().diagflat()
        if X2 is None:
            X1 = self.phi(X1)
            return X1.mm(variance.mm(X1.T))
        else:
            return self.phi(X1).mm(variance.mm(self.phi(X2).T))

class ExponentialKernel(Kernel):
    """
    An exponential kernel given by

    $$ K(x,x') = \\sigma^2 \\exp\\left(-\\frac{|x-x'|}{2l}\\right) $$

    with \\(\\sigma^2\\) the magnitude, \\(l\\) the lengthscale.

    Args:
        input_dims (int): Number of input dimensions.
        active_dims (list of int): Indices of active dimensions of shape (input_dims,).
        name (str): Kernel name.

    Attributes:
        magnitude (mogptk.gpr.parameter.Parameter): Magnitude \\(\\sigma^2\\) a scalar.
        lengthscale (mogptk.gpr.parameter.Parameter): Lengthscale \\(l\\) of shape (input_dims,).
    """
    def __init__(self, input_dims=1, active_dims=None, name="Exponential"):
        super().__init__(input_dims, active_dims, name)

        magnitude = 1.0
        lengthscale = torch.ones(input_dims)

        self.magnitude = Parameter(magnitude, lower=config.positive_minimum)
        self.lengthscale = Parameter(lengthscale, lower=config.positive_minimum)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1, X2 = self._active_input(X1, X2)
        dist = torch.abs(self.distance(X1,X2))  # NxMxD
        exp = -0.5*torch.tensordot(dist, 1.0/self.lengthscale(), dims=1)  # NxM
        return self.magnitude() * torch.exp(exp)

    def K_diag(self, X1):
        # X has shape (data_points,input_dims)
        X1, _ = self._active_input(X1)
        return self.magnitude().repeat(X1.shape[0])

class SquaredExponentialKernel(Kernel):
    """
    A squared exponential kernel given by

    $$ K(x,x') = \\sigma^2 \\exp\\left(-\\frac{1}{2}\\tau^T\\Sigma\\tau\\right) $$

    with \\(\\tau = |x-x'|\\), \\(\\Sigma = LL^T + \\mathrm{diag}(l)\\), \\(\\sigma^2\\) the magnitude, \\(l\\) the lengthscales, and \\(LL^T\\) the cross lengthscales.

    Args:
        order (float): Order of cross lengthscales.
        input_dims (int): Number of input dimensions.
        active_dims (list of int): Indices of active dimensions of shape (input_dims,).
        name (str): Kernel name.

    Attributes:
        magnitude (mogptk.gpr.parameter.Parameter): Magnitude \\(\\sigma^2\\) a scalar.
        lengthscale (mogptk.gpr.parameter.Parameter): Lengthscale \\(l\\) of shape (input_dims,).
        cross_lengthscale (mogptk.gpr.parameter.Parameter): Lower triangular cholesky of the cross lengthscale \\(L\\) of shape (input_dims,order).
    """
    def __init__(self, order=0, input_dims=1, active_dims=None, name="SE"):
        super().__init__(input_dims, active_dims, name)

        magnitude = 1.0
        lengthscale = torch.ones(input_dims)
        cross_lengthscale = torch.ones(input_dims,order)

        self.magnitude = Parameter(magnitude, lower=config.positive_minimum)
        self.lengthscale = Parameter(lengthscale, lower=config.positive_minimum)
        self.cross_lengthscale = Parameter(cross_lengthscale, lower=config.positive_minimum)
        self.cross_lengthscale.num_parameters = int((input_dims*input_dims+input_dims)/2)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1, X2 = self._active_input(X1, X2)
        dist = self.distance(X1,X2)  # NxMxD
        lengthscale = self.cross_lengthscale().tril().mm(self.cross_lengthscale().tril().T) + torch.diag(self.lengthscale()**2)  # DxD
        exp = -0.5*torch.einsum("ijk,kl,ijl->ij",dist,lengthscale,dist)  # NxM
        return self.magnitude() * torch.exp(exp)

    def K_diag(self, X1):
        # X has shape (data_points,input_dims)
        X1, _ = self._active_input(X1)
        return self.magnitude().repeat(X1.shape[0])

class RationalQuadraticKernel(Kernel):
    """
    A rational quadratic kernel given by

    $$ K(x,x') = \\sigma^2 \\left(1 + \\frac{|x-x'|^2}{2l^2}\\right)^{-\\alpha} $$

    with \\(\\sigma^2\\) the magnitude, \\(l\\) the lengthscale, and \\(\\alpha\\) the relative weighting of small-scale and large-scale fluctuations. When \\(\\alpha \\to \\infty\\) this kernel becomes equivalent to the squared exponential kernel.

    Args:
        input_dims (int): Number of input dimensions.
        active_dims (list of int): Indices of active dimensions of shape (input_dims,).
        name (str): Kernel name.

    Attributes:
        alpha (float): Relative weighting of small-scale and large-scale fluctuations \\(\\alpha\\).
        magnitude (mogptk.gpr.parameter.Parameter): Magnitude \\(\\sigma^2\\) a scalar.
        lengthscale (mogptk.gpr.parameter.Parameter): Lengthscale \\(l\\) of shape (input_dims,).
    """
    def __init__(self, alpha=1.0, input_dims=1, active_dims=None, name="RQ"):
        super().__init__(input_dims, active_dims, name)

        magnitude = 1.0
        lengthscale = torch.ones(input_dims)

        self.alpha = alpha
        self.magnitude = Parameter(magnitude, lower=config.positive_minimum)
        self.lengthscale = Parameter(lengthscale, lower=config.positive_minimum)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1, X2 = self._active_input(X1, X2)
        sqdist = self.squared_distance(X1,X2)  # NxMxD
        power = 1.0+0.5*torch.tensordot(sqdist, 1.0/self.lengthscale()**2, dims=1)/self.alpha  # NxM
        return self.magnitude() * torch.pow(power,-self.alpha)

    def K_diag(self, X1):
        # X has shape (data_points,input_dims)
        X1, _ = self._active_input(X1)
        return self.magnitude().repeat(X1.shape[0])

class PeriodicKernel(Kernel):
    """
    A periodic kernel given by

    $$ K(x,x') = \\sigma^2 \\exp\\left(-\\frac{2\\sin^2(\\pi |x-x'| / p)}{l^2}\\right) $$

    with \\(\\sigma^2\\) the magnitude, \\(l\\) the lengthscale, and \\(p\\) the period parameter.

    Args:
        input_dims (int): Number of input dimensions.
        active_dims (list of int): Indices of active dimensions of shape (input_dims,).
        name (str): Kernel name.

    Attributes:
        magnitude (mogptk.gpr.parameter.Parameter): Magnitude \\(\\sigma^2\\) a scalar.
        lengthscale (mogptk.gpr.parameter.Parameter): Lengthscale \\(l\\) of shape (input_dims,).
        period (mogptk.gpr.parameter.Parameter): Period \\(p\\) of shape (input_dims,).
    """
    def __init__(self, input_dims=1, active_dims=None, name="Periodic"):
        super().__init__(input_dims, active_dims, name)

        magnitude = 1.0
        lengthscale = torch.ones(input_dims)
        period = torch.ones(input_dims)

        self.magnitude = Parameter(magnitude, lower=config.positive_minimum)
        self.lengthscale = Parameter(lengthscale, lower=config.positive_minimum)
        self.period = Parameter(period, lower=config.positive_minimum)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1, X2 = self._active_input(X1, X2)
        tau = self.distance(X1,X2)
        sin = torch.sin(np.pi * tau / self.period())  # NxMxD
        exp = -2.0 * torch.tensordot(sin**2, 1.0/self.lengthscale()**2, dims=1)  # NxM
        return self.magnitude() * torch.exp(exp)

    def K_diag(self, X1):
        # X has shape (data_points,input_dims)
        X1, _ = self._active_input(X1)
        return self.magnitude().repeat(X1.shape[0])

class LocallyPeriodicKernel(Kernel):
    """
    A locally periodic kernel given by

    $$ K(x,x') = \\sigma^2 \\exp\\left(-\\frac{2\\sin^2(\\pi |x-x'| / p)}{l^2}\\right) \\exp\\left(-\\frac{|x-x'|^2}{2l^2}\\right) $$

    with \\(\\sigma^2\\) the magnitude, \\(l\\) the lengthscale, and \\(p\\) the period.

    Args:
        input_dims (int): Number of input dimensions.
        active_dims (list of int): Indices of active dimensions of shape (input_dims,).
        name (str): Kernel name.

    Attributes:
        magnitude (mogptk.gpr.parameter.Parameter): Magnitude \\(\\sigma^2\\) a scalar.
        lengthscale (mogptk.gpr.parameter.Parameter): Lengthscale \\(l\\) of shape (input_dims,).
        period (mogptk.gpr.parameter.Parameter): Period \\(p\\) of shape (input_dims,).
    """
    def __init__(self, input_dims=1, active_dims=None, name="LocallyPeriodic"):
        super().__init__(input_dims, active_dims, name)

        magnitude = 1.0
        lengthscale = torch.ones(input_dims)
        period = torch.ones(input_dims)

        self.magnitude = Parameter(magnitude, lower=config.positive_minimum)
        self.lengthscale = Parameter(lengthscale, lower=config.positive_minimum)
        self.period = Parameter(period, lower=config.positive_minimum)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1, X2 = self._active_input(X1, X2)
        tau = self.distance(X1,X2)
        sqdist = self.squared_distance(X1,X2)
        sin = torch.sin(np.pi * tau / self.period())  # NxMxD
        exp1 = -2.0 * torch.tensordot(sin**2, 1.0/self.lengthscale()**2, dims=1)  # NxM
        exp2 = -0.5 * torch.tensordot(sqdist, 1.0/self.lengthscale()**2, dims=1)  # NxM
        return self.magnitude() * torch.exp(exp1) * torch.exp(exp2)

    def K_diag(self, X1):
        # X has shape (data_points,input_dims)
        X1, _ = self._active_input(X1)
        return self.magnitude().repeat(X1.shape[0])

class CosineKernel(Kernel):
    """
    A cosine periodic kernel given by

    $$ K(x,x') = \\sigma^2 \\cos(2\\pi |x-x'| / l) $$

    with \\(\\sigma^2\\) the magnitude and \\(l\\) the lengthscale.

    Args:
        input_dims (int): Number of input dimensions.
        active_dims (list of int): Indices of active dimensions of shape (input_dims,).
        name (str): Kernel name.

    Attributes:
        magnitude (mogptk.gpr.parameter.Parameter): Magnitude \\(\\sigma^2\\) a scalar.
        lengthscale (mogptk.gpr.parameter.Parameter): Lengthscale \\(l\\) of shape (input_dims,).
    """
    def __init__(self, input_dims=1, active_dims=None, name="Cosine"):
        super().__init__(input_dims, active_dims, name)

        magnitude = 1.0
        lengthscale = torch.ones(input_dims)

        self.magnitude = Parameter(magnitude, lower=config.positive_minimum)
        self.lengthscale = Parameter(lengthscale, lower=config.positive_minimum)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1, X2 = self._active_input(X1, X2)
        tau = self.distance(X1,X2)
        cos = 2.0*np.pi * torch.tensordot(tau, 1.0/self.lengthscale(), dims=1)  # NxM
        return self.magnitude() * torch.cos(cos)

    def K_diag(self, X1):
        # X has shape (data_points,input_dims)
        X1, _ = self._active_input(X1)
        return self.magnitude().repeat(X1.shape[0])

class SincKernel(Kernel):
    """
    A sinc kernel given by

    $$ K(x,x') = \\sigma^2 \\frac{\\sin(\\Delta |x-x'|)}{\\Delta |x-x'|} \\cos(2\\pi \\xi_0 |x-x'|) $$

    with \\(\\sigma^2\\) the magnitude, \\(\\Delta\\) the bandwidth, and \\(\\xi_0\\) the frequency.

    Args:
        input_dims (int): Number of input dimensions.
        active_dims (list of int): Indices of active dimensions of shape (input_dims,).
        name (str): Kernel name.

    Attributes:
        magnitude (mogptk.gpr.parameter.Parameter): Magnitude \\(\\sigma^2\\) a scalar.
        frequency (mogptk.gpr.parameter.Parameter): Frequency \\(\\xi_0\\) of shape (input_dims,).
        bandwidth (mogptk.gpr.parameter.Parameter): Bandwidth \\(\\Delta\\) of shape (input_dims,).
    """
    def __init__(self, input_dims=1, active_dims=None, name="Sinc"):
        super().__init__(input_dims, active_dims, name)

        magnitude = 1.0
        frequency = torch.ones(input_dims)
        bandwidth = torch.ones(input_dims)

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
        return self.magnitude() * self._sinc(sinc) * torch.cos(cos)

    def K_diag(self, X1):
        # X has shape (data_points,input_dims)
        X1, _ = self._active_input(X1)
        return self.magnitude().repeat(X1.shape[0])

class SpectralKernel(Kernel):
    """
    A spectral kernel given by

    $$ K(x,x') = \\sigma^2 \\exp\\left(-2\\pi^2 \\Sigma |x-x'|^2\\right) \\cos(2\\pi \\mu |x-x'|) $$

    with \\(\\sigma^2\\) the magnitude, \\(\\Sigma\\) the variance, and \\(\\mu\\) the mean. When the mean is zero, this kernel is equivalent to the SquaredExponential kernel with \\(l = \\frac{1}{2\\pi\\sqrt{\\Sigma}}\\).

    Args:
        input_dims (int): Number of input dimensions.
        active_dims (list of int): Indices of active dimensions of shape (input_dims,).
        name (str): Kernel name.

    Attributes:
        magnitude (mogptk.gpr.parameter.Parameter): Magnitude \\(\\sigma^2\\) a scalar.
        mean (mogptk.gpr.parameter.Parameter): Mean \\(\\mu\\) of shape (input_dims,).
        variance (mogptk.gpr.parameter.Parameter): Variance \\(\\Sigma\\) of shape (input_dims,).
    """
    def __init__(self, input_dims=1, active_dims=None, name="Spectral"):
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
        return self.magnitude() * torch.prod(torch.exp(exp) * torch.cos(cos), dim=2)

    def K_diag(self, X1):
        # X has shape (data_points,input_dims)
        X1, _ = self._active_input(X1)
        return self.magnitude().repeat(X1.shape[0])

class SpectralMixtureKernel(Kernel):
    """
    A spectral mixture kernel given by

    $$ K(x,x') = \\sum_{q=0}^Q \\sigma_q^2 \\exp\\left(-2\\pi^2 \\Sigma_q |x-x'|^2\\right) \\cos(2\\pi \\mu_q |x-x'|) $$

    with \\(Q\\) the number of mixtures, \\(\\sigma^2\\) the magnitude, \\(\\Sigma\\) the variance, and \\(\\mu\\) the mean.

    Args:
        Q (int): Number of mixture components.
        input_dims (int): Number of input dimensions.
        active_dims (list of int): Indices of active dimensions of shape (input_dims,).
        name (str): Kernel name.

    Attributes:
        magnitude (mogptk.gpr.parameter.Parameter): Magnitude \\(\\sigma^2\\) of shape (Q,).
        mean (mogptk.gpr.parameter.Parameter): Mean \\(\\mu\\) of shape (Q,input_dims).
        variance (mogptk.gpr.parameter.Parameter): Variance \\(\\Sigma\\) of shape (Q,input_dims).
    """
    def __init__(self, Q=1, input_dims=1, active_dims=None, name="SM"):
        super().__init__(input_dims, active_dims, name)

        magnitude = torch.ones(Q)
        mean = torch.zeros(Q,input_dims)
        variance = torch.ones(Q,input_dims)

        self.magnitude = Parameter(magnitude, lower=config.positive_minimum)
        self.mean = Parameter(mean, lower=config.positive_minimum)
        self.variance = Parameter(variance, lower=config.positive_minimum)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1, X2 = self._active_input(X1, X2)
        tau = self.distance(X1,X2)[None,:,:,:]  # 1xNxMxD
        exp = -2.0*np.pi**2 * tau**2 * self.variance()[:,None,None,:]  # QxNxMxD
        cos = 2.0*np.pi * tau * self.mean()[:,None,None,:]  # QxNxMxD
        Kq = self.magnitude()[:,None,None] * torch.prod(torch.exp(exp) * torch.cos(cos), dim=3)
        return torch.sum(Kq, dim=0)

    def K_diag(self, X1):
        # X has shape (data_points,input_dims)
        X1, _ = self._active_input(X1)
        return torch.sum(self.magnitude()).repeat(X1.shape[0])

class MaternKernel(Kernel):
    """
    A Matérn kernel given by

    $$ K(x,x') = \\sigma^2 c \\exp\\left(-\\sqrt{2\\nu |x-x'| / l}\\right) $$

    with \\(\\sigma^2\\) the magnitude, \\(l\\) the lengthscale, and \\(c\\) depending on \\(\\nu\\) is either \\(1.0\\) for \\(\\nu = 0.5\\), or \\(1.0 + \\sqrt{3}|x-x'|/l\\) for \\(\\nu = 1.5\\), or \\(1.0 + \\sqrt{5}|x-x'|/l + \\frac{5|x-x'|^2}{3l^2}\\).

    Args:
        nu (float): Parameter that must be 0.5, 1.5, or 2.5.
        input_dims (int): Number of input dimensions.
        active_dims (list of int): Indices of active dimensions of shape (input_dims,).
        name (str): Kernel name.

    Attributes:
        magnitude (mogptk.gpr.parameter.Parameter): Magnitude \\(\\sigma^2\\) a scalar.
        lengthscale (mogptk.gpr.parameter.Parameter): Lengthscale \\(l\\) of shape (input_dims,).
    """
    def __init__(self, nu=0.5, input_dims=1, active_dims=None, name="Matérn"):
        super().__init__(input_dims, active_dims, name)

        if nu not in [0.5, 1.5, 2.5]:
            raise ValueError("nu parameter must be 0.5, 1.5, or 2.5")

        magnitude = 1.0
        lengthscale = torch.ones(input_dims)

        self.nu = nu
        self.magnitude = Parameter(magnitude, lower=1e-6)
        self.lengthscale = Parameter(lengthscale, lower=1e-6)

    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1, X2 = self._active_input(X1, X2)
        if X2 is None:
            X2 = X1

        dist = torch.abs(torch.tensordot(self.distance(X1,X2), 1.0/self.lengthscale(), dims=1))
        if self.nu == 0.5:
            constant = 1.0
        elif self.nu == 1.5:
            constant = 1.0 + np.sqrt(3.0)*dist
        elif self.nu == 2.5:
            constant = 1.0 + np.sqrt(5.0)*dist + 5.0/3.0*dist**2
        return self.magnitude() * constant * torch.exp(-np.sqrt(self.nu*2.0) * dist)

    def K_diag(self, X1):
        # X has shape (data_points,input_dims)
        X1, _ = self._active_input(X1)
        return self.magnitude().repeat(X1.shape[0])
