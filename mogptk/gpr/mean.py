import torch
from . import Parameter, config

class Mean(torch.nn.Module):
    """
    Defines a trainable mean function, complementary to the the way we also have a trainable covariance function (the kernel).
    """
    def __init__(self):
        super().__init__()

    def __call__(self, X):
        """
        Return the mean for a given `X`. This is the same as calling `mean(X)` but `X` doesn't necessarily have to be a tensor.

        Args:
            X (torch.tensor): Input of shape (data_points,input_dims).

        Returns:
            torch.tensor: Mean values of shape (data_points,).
        """
        X = self._check_input(X)
        return self.mean(X)

    def __setattr__(self, name, val):
        if name == 'train':
            for p in self.parameters():
                p.train = val
            return
        if hasattr(self, name) and isinstance(getattr(self, name), Parameter):
            raise AttributeError("parameter is read-only, use Parameter.assign()")
        if isinstance(val, Parameter) and val._name is None:
            val._name = 'mean.' + name
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
        """
        Return the mean for a given `X`.

        Args:
            X (torch.tensor): Input of shape (data_points,input_dims).

        Returns:
            torch.tensor: Mean values of shape (data_points,).
        """
        raise NotImplementedError()

class ConstantMean(Mean):
    """
    Constant mean function:

    $$ m(X) = b $$

    with \\(b\\) the bias.

    Args:

    Attributes:
        bias (mogptk.gpr.parameter.Parameter): Bias \\(b\\).
    """
    def __init__(self):
        super().__init__()

        self.bias = Parameter(0.0)

    def mean(self, X):
        return self.bias().repeat(X.shape[0])

class LinearMean(Mean):
    """
    Linear mean function:

    $$ m(X) = aX + b $$

    with \\(a\\) the slope and \\(b\\) the bias.

    Args:
        input_dims (int): Number of input dimensions.

    Attributes:
        bias (mogptk.gpr.parameter.Parameter): Bias \\(b\\).
        slope (mogptk.gpr.parameter.Parameter): Slope \\(a\\).
    """
    def __init__(self, input_dims=1):
        super().__init__()

        self.bias = Parameter(0.0)
        self.slope = Parameter(torch.zeros(input_dims))

    def mean(self, X):
        return self.bias() + X.mm(self.slope().reshape(-1,1))
