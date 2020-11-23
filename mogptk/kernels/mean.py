from . import Parameter

class Mean:
    def __init__(self, name=None):
        if name is None:
            name = self.__class__.__name__
            if name.endswith('Mean') and name != 'Mean':
                name = name[:-4]
        self.name = name

    def __call__(self, X):
        raise NotImplementedError()

    def __setattr__(self, name, val):
        if name == 'trainable':
            from .util import _find_parameters
            for p in _find_parameters(self):
                p.trainable = val
            return
        if hasattr(self, name) and isinstance(getattr(self, name), Parameter):
            raise AttributeError("parameter is read-only, use Parameter.assign()")
        if isinstance(val, Parameter) and val.name is None:
            val.name = name
        super(Mean,self).__setattr__(name, val)        
