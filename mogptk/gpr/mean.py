import torch
from . import Parameter, config

class Mean:
    def __init__(self, name="Mean"):
        self.name = name

    def __call__(self, X):
        X = self._check_input(X)
        return self.mean(X)

    def __setattr__(self, name, val):
        if name == 'trainable':
            from .util import _find_parameters
            for _, p in _find_parameters(self):
                p.trainable = val
            return
        if hasattr(self, name) and isinstance(getattr(self, name), Parameter):
            raise AttributeError("parameter is read-only, use Parameter.assign()")
        if isinstance(val, Parameter) and val.name is None:
            val.name = name
        super().__setattr__(name, val)

    def _check_input(self, X):
        if not torch.is_tensor(X):
            X = torch.tensor(X, device=config.device, dtype=config.dtype)
        elif X.device != config.device or X.dtype != config.dtype:
            X = X.to(device, dtype)
        if X.ndim != 2:
            raise ValueError("X should have two dimensions (data_points,input_dims)")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must not be empty")
        return X

    def mean(self, X):
        raise NotImplementedError()

class ConstantMean(Mean):
    def __init__(self, name="ConstantMean"):
        super().__init__(name)
        self.bias = Parameter(0.0)

    def mean(self, X):
        return self.bias().repeat(X.shape[0])

class LinearMean(Mean):
    def __init__(self, input_dims=1, name="LinearMean"):
        super().__init__(name)
        self.bias = Parameter(0.0)
        self.slope = Parameter(torch.zeros(input_dims))

    def mean(self, X):
        return self.bias() + X.mm(self.slope().reshape(1,-1))
