import copy
import numpy as np

class Transformer:
    def __init__(self, transformers=None):
        if transformers is None:
            transformers = []
        if not isinstance(transformers, list):
            transformers = [transformers]
        if not all(issubclass(type(t), TransformBase) for t in transformers):
            raise ValueError('transformer must derive from TransformBase')
        self.transformers = transformers

    def append(self, t, y, x=None):
        if isinstance(t, type):
            t = t()
        else:
            t = copy.deepcopy(t)
        y = self.forward(y, x)
        t.set_data(y, x)
        self.transformers.append(t)

    def forward(self, y, x=None):
        for t in self.transformers:
            y = t.forward(y, x)
        return y

    def backward(self, y, x=None):
        for t in self.transformers[::-1]:
            y = t.backward(y, x)
        return y

class TransformBase:
    """
    TransformBase is a base class for transformations. Each derived class must at least implement the `forward()` and `backward()` functions.

    """
    def set_data(self, y, x=None):
        pass

    def forward(self, y, x=None):
        raise NotImplementedError
    
    def backward(self, y, x=None):
        raise NotImplementedError

class TransformDetrend(TransformBase):
    """
    TransformDetrend is a transformer that detrends the data. It uses `numpy.polyfit` to find a polynomial of given degree that best fits the data and thus removes the trend.

    Args:
        degree (int): Polynomial degree that will be fit, i.e. `2` will find a quadratic trend and remove it from the data.
        input_dim (int): Input dimension to operate on.
    """
    # TODO: add regression?
    def __init__(self, degree=1, input_dim=0):
        self.degree = degree
        self.dim = input_dim

    def __repr__(self):
        return 'TransformDetrend(degree=%g)' % (self.degree,)

    def set_data(self, y, x=None):
        self.coef = np.polyfit(x[:,self.dim], y, self.degree)

    def forward(self, y, x):
        if x is None:
            raise ValueError("must set X for transformation")
        x = x[:,self.dim]
        return y - np.polyval(self.coef, x)
    
    def backward(self, y, x):
        if x is None:
            raise ValueError("must set X for transformation")
        x = x[:,self.dim]
        return y + np.polyval(self.coef, x)

class TransformLinear(TransformBase):
    """
    TransformLinear transforms the data linearly so that y => (y-bias)/slope.
    """
    def __init__(self, bias=0.0, slope=1.0):
        self.bias = bias
        self.slope = slope

    def __repr__(self):
        return 'TransformLinear(bias=%g, slope=%g)' % (self.bias, self.slope)

    def forward(self, y, x=None):
        return (y-self.bias)/self.slope

    def backward(self, y, x=None):
        return self.bias + self.slope*y

class TransformNormalize(TransformBase):
    """
    TransformNormalize is a transformer that normalizes the data so that the Y axis is between -1 and 1.
    """
    def __init__(self):
        pass

    def __repr__(self):
        return 'TransformNormalize(min=%g, max=%g)' % (self.ymin, self.ymax)

    def set_data(self, y, x=None):
        self.ymin = np.amin(y)
        self.ymax = np.amax(y)

    def forward(self, y, x=None):
        return -1.0 + 2.0*(y-self.ymin)/(self.ymax-self.ymin)
    
    def backward(self, y, x=None):
        return (y+1.0)/2.0*(self.ymax-self.ymin)+self.ymin

class TransformLog(TransformBase):
    """
    TransformLog is a transformer that takes the log of the data. Data is automatically shifted in the Y axis so that all values are greater than or equal to 1.
    """
    def __init__(self):
        pass

    def __repr__(self):
        return 'TransformLog(shift=%g, mean=%g)' % (self.shift, self.mean)

    def set_data(self, y, x=None):
        self.shift = 1 - y.min()
        self.mean = np.log(y + self.shift).mean()

    def forward(self, y, x=None):
        return np.log(y + self.shift) - self.mean
    
    def backward(self, y, x=None):
        return np.exp(y + self.mean) - self.shift

class TransformStandard(TransformBase):
    """
    TransformStandard is a transformer that whitens the data. That is, it transform the data so it has zero mean and unit variance.
    """
    def __init__(self):
        pass

    def __repr__(self):
        return 'TransformStandard(mean=%g, std=%g)' % (self.mean, self.std)
    
    def set_data(self, y, x=None):
        self.mean = y.mean()
        self.std = y.std()
        
    def forward(self, y, x=None):
        return (y - self.mean) / self.std
    
    def backward(self, y, x=None):
        return (y * self.std) + self.mean
