import torch
import torch.nn.functional as functional

from .config import config

class Transform:
    def forward(self, x):
        # unconstrained to constrained space
        raise NotImplementedError()
    
    def inverse(self, y):
        # constrained to unconstrained space
        raise NotImplementedError()

class Softplus(Transform):
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
    def __init__(self, lower=0.0, upper=1.0):
        self.lower = lower
        self.upper = upper

    def forward(self, x):
        return self.lower + (self.upper-self.lower)*torch.sigmoid(x)

    def inverse(self, y):
        if torch.any(y < self.lower) or torch.any(self.upper < y):
            raise ValueError("values must be between %s and %s" % (self.lower, self.upper))
        y = y/(self.upper-self.lower) - self.lower
        return torch.log(y) - torch.log(1-y)

class Parameter:
    def __init__(self, data, name=None, lower=None, upper=None, prior=None, trainable=True):
        self.name = name
        self.lower = None
        self.upper = None
        self.prior = prior
        self.trainable = trainable
        self.transform = None
        self.unconstrained = None
        self.assign(data, lower=lower, upper=upper)

    def __repr__(self):
        if self.name is None:
            return '{}'.format(self.constrained.tolist())
        return '{}={}'.format(self.name, self.constrained.tolist())
    
    def __call__(self):
        return self.constrained

    @property
    def constrained(self):
        if self.transform is not None:
            return self.transform.forward(self.unconstrained)
        return self.unconstrained

    def numpy(self):
        return self.constrained.detach().numpy()

    def assign(self, data, name=None, lower=None, upper=None, prior=None, trainable=None):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, device=config.device, dtype=config.dtype)
        else:
            data = data.to(config.device, config.dtype)
        if lower is not None:
            lower = torch.tensor(lower, device=config.device, dtype=config.dtype)
            if len(lower.shape) != 0 and lower.shape != data.shape:
                raise ValueError("lower and data must match shapes: %s != %s" % (lower.shape, data.shape))
        if upper is not None:
            upper = torch.tensor(upper, device=config.device, dtype=config.dtype)
            if len(upper.shape) != 0 and upper.shape != data.shape:
                raise ValueError("upper and data must match shapes: %s != %s" % (upper.shape, data.shape))
        if data.requires_grad:
            data = data.detach()

        if self.unconstrained is not None:
            origshape = data.shape
            while len(data.shape) < len(self.unconstrained.shape) and self.unconstrained.shape[len(data.shape)] == 1:
                data = data.unsqueeze(len(data.shape))
            if data.shape != self.unconstrained.shape:
                raise ValueError("parameter shape must match: %s != %s" % (origshape, self.unconstrained.shape))

        if name is None:
            name = self.name
        if lower is None:
            lower = self.lower
        if upper is None:
            upper = self.upper
        if prior is None:
            prior = self.prior
        if trainable is None:
            trainable = self.trainable

        transform = None
        if lower is not None and upper is not None:
            lower_tmp = torch.min(lower, upper)
            upper = torch.max(lower, upper)
            lower = lower_tmp
            transform = Sigmoid(lower=lower, upper=upper)
        elif lower is not None:
            transform = Softplus(lower=lower)
        elif upper is not None:
            transform = Softplus(lower=upper, beta=-1.0)

        if transform is not None:
            if lower is not None:
                data = torch.where(data < lower, lower * torch.ones_like(data), data)
            if upper is not None:
                data = torch.where(upper < data, upper * torch.ones_like(data), data)
            data = transform.inverse(data)
        data.requires_grad = True

        self.name = name
        self.prior = prior
        self.lower = lower
        self.upper = upper
        self.trainable = trainable
        self.transform = transform
        self.unconstrained = data

    def log_prior(self):
        if self.prior is None:
            return 0.0
        return self.prior.log_p(self()).sum()
