import numpy as np

class TransformBase:
    def set_data(self, data):
        pass

    def forward(self, y, x=None):
        raise NotImplementedError
    
    def backward(self, y, x=None):
        raise NotImplementedError

class Serie(np.ndarray):
    def __new__(cls, array, transformers=[], x=None, transformed=None):
        array = np.asarray(array)
        if array.ndim != 1 or array.shape[0] == 0:
            raise ValueError('Serie must have one dimension and a length greater than zero')
        elif not np.issubdtype(array.dtype, np.number) and not np.issubdtype(array.dtype, np.datetime64):
            raise ValueError('Serie must have a number of datetime64 data type')

        obj = np.asarray(array).view(cls)
        if transformed is None:
            obj.transformed = array.astype(np.float64) # may through exception for bad dtypes
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
        self.transformed = getattr(obj, 'transformed', None)
        self.transformers = getattr(obj, 'transformers', None)
    
    def __getitem__(self, index):
        ret = super(Serie, self).__getitem__(index)
        if isinstance(ret, Serie):
            ret.transformed = ret.transformed.__getitem__(index)
        return ret

    def apply(self, transformers, x=None):
        if not isinstance(transformers, list):
            transformers = [transformers]
        if not all(issubclass(type(t), TransformBase) for t in transformers):
            raise ValueError('transformers must derive from TransformBase')
        if x is not None and (x.ndim != 2 or x.shape[0] != self.shape[0]):
            raise ValueError('x must have two dimensions and a length equal to the series')

        for t in transformers:
            self.transformed = np.array(t.forward(self.transformed, x))
            self.transformers.append(t)

    def get_time_unit(self):
        if not np.issubdtype(self.dtype, np.datetime64):
            raise ValueError('Serie must have a datetime64 data type')

        unit = str(self.dtype)
        locBracket = unit.find('[')
        if locBracket == -1:
            return ''
        return unit[locBracket+1:-1]

    def set_time_unit(self, unit):
        if not np.issubdtype(self.dtype, np.datetime64):
            raise ValueError('Serie must have a datetime64 data type')
        self.astype('datetime64[%s]' % (unit,))

    def get_transformed(self):
        return self.transformed

    def transform(self, array, x=None):
        array = array.astype(self.dtype).astype(np.float64)
        for t in self.transformers:
            array = np.array(t.forward(array, x))
        return array

    def detransform(self, array, x=None):
        for t in self.transformers[::-1]:
            array = np.array(t.backward(array, x))
        return array.astype(self.dtype)
