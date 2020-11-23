from . import Parameter, Mean, Kernel

def _find_parameters(obj):
    if isinstance(obj, Parameter):
        yield obj
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _find_parameters(v)
    elif issubclass(type(obj), Kernel) or issubclass(type(obj), Mean):
        for v in obj.__dict__.values():
            yield from _find_parameters(v)

