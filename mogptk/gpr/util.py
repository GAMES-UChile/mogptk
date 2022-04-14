import inspect
import torch

from .config import *
from .parameter import Parameter
from .model import Mean
from .kernel import Kernel
from .likelihood import Likelihood

def _find_parameters(obj, names=[]):
    if isinstance(obj, Parameter):
        yield names, obj
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _find_parameters(v, names + [i])
    elif issubclass(type(obj), (Kernel, Mean, Likelihood)):
        for i, v in obj.__dict__.items():
            yield from _find_parameters(v, names + [i])

def _copy_parameters(obj, other):
    if not type(obj) is type(other):
        raise ValueError("other must be of type %s" % (type(self),))

    for names, p in _find_parameters(obj):
        po = other
        for name in names:
            if not isinstance(po, list):
                po = po.__dict__
            po = po[name]
        p.assign(po)

def merge_data(xs, ys=None):
    if not isinstance(xs, list) or ys is not None and not isinstance(ys, list):
        raise ValueError("input must be a list of channels")
    output_dims = len(xs)
    if ys is not None and len(ys) != output_dims:
        raise ValueError("inputs must have the same number of output dimensions")
    N = [x.shape[0] for x in xs]
    if ys is not None and not all(y.shape[0] == N[i] for i, y in enumerate(ys)):
        raise ValueError("inputs must have the same number of data points per output dimension")

    n = 0
    X = torch.zeros((sum(N),1+xs[0].shape[1]), device=config.device, dtype=config.dtype)
    if ys is not None:
        Y = torch.zeros((sum(N),1), device=config.device, dtype=config.dtype)
    for channel, x in enumerate(xs):
        X[n:n+N[channel],0] = channel
        X[n:n+N[channel],1:] = x
        if ys is not None:
            Y[n:n+N[channel],:] = ys[channel]
        n += N[channel]
    if ys is not None:
        return N, X, Y
    return N, X

def split_data(N, *Xs):
    if not all(X.ndim == 2 for X in Xs):
        raise ValueError("inputs must have shape (data_points,dimensions)")
    if not all(X.shape[0] == sum(N) for X in Xs):
        raise ValueError("inputs must have number of data points that correspond to N")

    data = []
    for X in Xs:
        n = 0
        xs = []
        for channel in range(len(N)):
            xs.append(X[n:n+N[channel],:])
            n += N[channel]
        data.append(xs)
    if len(data) == 1:
        return data[0]
    return (*data,)
