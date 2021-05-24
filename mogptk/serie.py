import copy
import numpy as np

class TransformBase:
    """
    TransformBase is a base class for transformations. Each derived class must at least implement the `forward()` and `backward()` functions.

    """
    def set_data(self, data):
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
    """
    # TODO: add regression?
    def __init__(self, degree=1, dim=0):
        self.degree = degree
        self.dim = dim

    def __repr__(self):
        return 'TransformDetrend(degree=%g)' % (self.degree,)

    def set_data(self, data):
        if data.get_input_dims() != 1:
            raise ValueError("can only remove ranges on one dimensional input data")

        x = np.array(data.X[0][data.mask]).astype(np.float64)
        self.coef = np.polyfit(x, data.Y.transformed[data.mask], self.degree)

    def forward(self, y, x):
        if x is None:
            raise ValueError("must set X for transformation")
        x = x[self.dim].astype(np.float64)
        return y - np.polyval(self.coef, x)
    
    def backward(self, y, x):
        if x is None:
            raise ValueError("must set X for transformation")
        x = x[self.dim].astype(np.float64)
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

    def set_data(self, data):
        pass

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

    def set_data(self, data):
        self.ymin = np.amin(data.Y.transformed[data.mask])
        self.ymax = np.amax(data.Y.transformed[data.mask])

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

    def set_data(self, data):
        self.shift = 1 - data.Y.transformed.min()
        self.mean = np.log(data.Y.transformed + self.shift).mean()

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
    
    def set_data(self, data):
        # take only the non-removed observations
        self.mean = data.Y.transformed[data.mask].mean()
        self.std = data.Y.transformed[data.mask].std()
        
    def forward(self, y, x=None):
        return (y - self.mean) / self.std
    
    def backward(self, y, x=None):
        return (y * self.std) + self.mean

# TODO: add TransformFunction with a given mean function

################################################################
################################################################
################################################################

class Serie(np.ndarray):
    """
    Serie is an extension to the `numpy.ndarray` data type and includes transformations for the array. That is, it maintains the original data array, but also keeps a transformed data array to improve training using e.g. Gaussian processes. By storing the chain of transformations, it is possible to detransform predictions done in the transformed space to the original space and thus allows analysis or plotting in the original domain. Automatic conversions is performed for `numpy.datetime64` arrays to `numpy.float` arrays.
    """
    def __new__(cls, array, transformers=[], x=None, transformed=None):
        array = np.asarray(array)
        #if array.ndim != 1 or array.shape[0] == 0:
        #    raise ValueError('Serie must have one dimension and a length greater than zero')
        if not np.issubdtype(array.dtype, np.float64) and not np.issubdtype(array.dtype, np.datetime64):
            raise ValueError('Serie must have a float64 or datetime64 data type')

        obj = np.asarray(array).view(cls)
        if transformed is None:
            obj.transformed = array.astype(np.float64)
            obj.transformers = []
            obj.apply(transformers, x)
        else:
            obj.transformed = transformed
            obj.transformers = obj.transformers

        obj.flags['WRITEABLE'] = False
        obj.transformed.flags['WRITEABLE'] = False
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.transformed = copy.deepcopy(getattr(obj, 'transformed', None))
        self.transformers = copy.deepcopy(getattr(obj, 'transformers', None))
    
    def __getitem__(self, index):
        ret = super(Serie, self).__getitem__(index)
        if isinstance(ret, Serie):
            ret.transformed = ret.transformed.__getitem__(index)
        return ret

    def __reduce__(self):
        parent = super(Serie, self).__reduce__()
        return (parent[0], parent[1], parent[2] + (self.transformed,self.transformers))

    def __setstate__(self, data):
        super(Serie, self).__setstate__(data[:-2])
        self.transformed = data[-2]
        self.transformers = data[-1]

    def apply(self, transformers, x=None):
        if not isinstance(transformers, list):
            transformers = [transformers]
        if not all(issubclass(type(t), TransformBase) for t in transformers):
            raise ValueError('transformers must derive from TransformBase')
        #if x is not None and (x.ndim != 2 or x.shape[0] != self.shape[0]):
        #    raise ValueError('x must have two dimensions and a length equal to the series')

        for t in transformers:
            self.transformed = np.array(t.forward(self.transformed, x))
            self.transformers.append(t)

    def is_datetime64(self):
        return np.issubdtype(self.dtype, np.datetime64)

    def get_time_unit(self):
        if not self.is_datetime64():
            raise ValueError('Serie must have a datetime64 data type')

        unit = str(self.dtype)
        locBracket = unit.find('[')
        if locBracket == -1:
            return ''
        return unit[locBracket+1:-1]

    def transform(self, array, x=None):
        array = array.astype(self.dtype).astype(np.float64)
        for t in self.transformers:
            array = np.array(t.forward(array, x))
        return array

    def detransform(self, array, x=None):
        for t in self.transformers[::-1]:
            array = np.array(t.backward(array, x))
        return array.astype(self.dtype)
