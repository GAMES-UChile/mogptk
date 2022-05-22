import numpy as np
import torch
import torch.nn.functional as functional

from .config import config

class Transform:
    """
    Base transformation class for constrained parameter.
    """
    def forward(self, x):
        """
        Forward transformation from the unconstrained original space to the new constrained transformed space.
        """
        # unconstrained to constrained space
        raise NotImplementedError()
    
    def inverse(self, y):
        """
        Inverse transformation from the new constrained transformed space to the unconstrained original space.
        """
        # constrained to unconstrained space
        raise NotImplementedError()

class Softplus(Transform):
    """
    Softplus transformation for a lower constraint given by

    $$ y = a + \\frac{\\log\\left(1+e^{\\beta x}\\right)}{\\beta} $$

    with \\(a\\) the lower limit and \\(\\beta\\) the slope.

    Args:
        lower (float,torch.tensor): Lower limit.
        beta (float): Slope.
        threshold (float): Location from where to approximate the softplus by a linear function to avoid numerical problems.
    """
    def __init__(self, lower=0.0, beta=0.1, threshold=20.0):
        self.beta = beta
        self.lower = lower
        self.threshold = threshold

    def forward(self, x):
        return self.lower + functional.softplus(x, beta=self.beta, threshold=self.threshold)

    def inverse(self, y):
        if torch.any(y < self.lower):
            raise ValueError("values must be at least %s" % self.lower)
        return y-self.lower + torch.log(-torch.expm1(-self.beta*(y-self.lower)))/self.beta

class Sigmoid(Transform):
    """
    Sigmoid transformation for a lower and upper constraint given by

    $$ y = a + (b-1)\\frac{1}{1+e^{-x}} $$

    with \\(a\\) the lower limit and \\(b\\) the upper limit.

    Args:
        lower (float,torch.tensor): Lower limit.
        upper (float,torch.tensor): Upper limit.
    """
    def __init__(self, lower=0.0, upper=1.0):
        self.lower = lower
        self.upper = upper

    def forward(self, x):
        return self.lower + (self.upper-self.lower)*torch.sigmoid(x)

    def inverse(self, y):
        if torch.any(y < self.lower) or torch.any(self.upper < y):
            raise ValueError("values must be between %s and %s" % (self.lower, self.upper))
        y = (y-self.lower)/(self.upper-self.lower)
        return torch.log(y) - torch.log(1-y)

class Parameter:
    """
    Parameter class that allows for parameter training in a constraint space.

    Args:
        value (torch.tensor): Parameter (initial) value in the constraint space.
        name (str): Name.
        lower (float,torch.tensor): Lower limit.
        upper (float,torch.tensor): Upper limit.
        prior (Likelihood,torch.distribution): Prior distribution.
        train (boolean): Train parameter, otherwise keep its value fixed.
    """
    def __init__(self, value, name=None, lower=None, upper=None, prior=None, train=True):
        self.name = name
        self.lower = None
        self.upper = None
        self.prior = prior
        self.train = train
        self.transform = None
        self.unconstrained = None
        self.pegged_parameter = None
        self.pegged_transform = None

        self.assign(value, lower=lower, upper=upper)

        self.num_parameters = int(np.prod(self.unconstrained.shape))

    def __repr__(self):
        name = self.name
        if self.pegged:
            name = self.pegged_parameter.name
        if name is None:
            return '{}'.format(self.constrained.tolist())
        return '{}={}'.format(self.name, self.constrained.tolist())
    
    def __call__(self):
        """
        Get (constraint) parameter value.

        Returns:
            torch.tensor
        """
        return self.constrained

    @property
    def pegged(self):
        return self.pegged_parameter is not None

    @property
    def constrained(self):
        """
        Get (constraint) parameter value. Equivalent to calling `parameter()`.

        Returns:
            torch.tensor
        """
        if self.pegged:
            other = self.pegged_parameter.constrained
            if self.pegged_transform is not None:
                other = self.pegged_transform(other)
            return other
        if self.transform is not None:
            return self.transform.forward(self.unconstrained)
        return self.unconstrained

    def numpy(self):
        """
        Get NumPy representation of the (constraint) parameter value.

        Returns:
            numpy.ndarray: Parameter value.
        """
        return self.constrained.detach().cpu().numpy()

    @staticmethod
    def to_tensor(value):
        if isinstance(value, Parameter):
            value = value.constrained.detach()
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, device=config.device, dtype=config.dtype)
        else:
            value = value.detach().to(config.device, config.dtype)
        return value

    def assign(self, value=None, name=None, lower=None, upper=None, prior=None, train=None):
        """
        Assign a new value to the parameter. If any of the arguments is not passed, the current value will be kept.

        Args:
            value (torch.tensor): Parameter value in the constraint space.
            name (str): Name.
            lower (float,torch.tensor): Lower limit.
            upper (float,torch.tensor): Upper limit.
            prior (Likelihood,torch.distribution): Prior distribution.
            train (boolean): Train parameter, otherwise keep its value fixed.
        """
        if value is not None:
            value = Parameter.to_tensor(value)
            if self.unconstrained is not None:
                origshape = value.shape
                while value.ndim < self.unconstrained.ndim and self.unconstrained.shape[value.ndim] == 1:
                    value = value.unsqueeze(-1)
                while self.unconstrained.ndim < value.ndim and value.shape[-1] == 1:
                    value = value.squeeze(-1)
                if value.shape != self.unconstrained.shape:
                    raise ValueError("parameter shape must match: %s != %s" % (origshape, self.unconstrained.shape))
        else:
            value = self.constrained.detach()

        if lower is not None:
            if not isinstance(lower, torch.Tensor):
                lower = torch.tensor(lower, device=config.device, dtype=config.dtype)
            else:
                lower = lower.detach().to(config.device, config.dtype)

            if lower.ndim != 0:
                while lower.ndim < value.ndim and value.shape[lower.ndim] == 1:
                    lower = lower.unsqueeze(-1)
                while value.ndim < lower.ndim and lower.shape[-1] == 1:
                    lower = lower.squeeze(-1)
                if lower.shape != value.shape:
                    raise ValueError("lower and value must match shapes: %s != %s" % (lower.shape, value.shape))
        else:
            lower = self.lower

        if upper is not None:
            if not isinstance(upper, torch.Tensor):
                upper = torch.tensor(upper, device=config.device, dtype=config.dtype)
            else:
                upper = upper.detach().to(config.device, config.dtype)

            if upper.ndim != 0:
                while upper.ndim < value.ndim and value.shape[upper.ndim] == 1:
                    upper = upper.unsqueeze(-1)
                while value.ndim < upper.ndim and upper.shape[-1] == 1:
                    upper = upper.squeeze(-1)
                if upper.shape != value.shape:
                    raise ValueError("upper and value must match shapes: %s != %s" % (upper.shape, value.shape))
        else:
            upper = self.upper

        if name is None:
            name = self.name
        else:
            idx = self.name.rfind('.')
            if idx != -1:
                name = self.name[:idx+1] + name
        if prior is None:
            prior = self.prior
        if train is None:
            if self.pegged:
                train = True
            else:
                train = self.train

        transform = None
        if lower is not None and upper is not None:
            if torch.any(upper < lower):
                raise ValueError("lower limit %s must be lower than upper limit %s" % (lower, upper))
            transform = Sigmoid(lower=lower, upper=upper)
        elif lower is not None:
            transform = Softplus(lower=lower)
        elif upper is not None:
            transform = Softplus(lower=upper, beta=-1.0)

        if transform is not None:
            if lower is not None:
                value = torch.where(value < lower, lower * torch.ones_like(value), value)
            if upper is not None:
                value = torch.where(upper < value, upper * torch.ones_like(value), value)
            value = transform.inverse(value)
        value.requires_grad = True

        self.name = name
        self.prior = prior
        self.lower = lower
        self.upper = upper
        self.train = train
        self.transform = transform
        self.unconstrained = value
        self.pegged_parameter = None
        self.pegged_transform = None

    def peg(self, other, transform=None):
        """
        Peg parameter to other parameter. It will follow another parameter's value and will not be trained independently. Additionally it is possible to transform the value while pegging, that is parameter A can be equal to two times parameter B for example.

        Args:
            other (Parameter): The other parameter to which this parameter will be pegged.
            transform (function): Transformation from the other parameter to this one.
        """
        if not isinstance(other, Parameter):
            raise ValueError("parameter must be pegged to other parameter object")
        elif other.pegged:
            raise ValueError("cannot peg parameter to another pegged parameter")
        self.pegged_parameter = other
        self.pegged_transform = transform
        self.train = False

    def log_prior(self):
        """
        Get the log of the prior.

        Returns:
            float: Log prior.
        """
        if self.prior is None:
            return 0.0
        return self.prior.log_prob(self()).sum()
