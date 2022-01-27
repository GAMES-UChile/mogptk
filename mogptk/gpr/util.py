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

class GaussHermiteQuadrature:
    def __init__(self, deg=20, t_scale=None, w_scale=None):
        t, w = np.polynomial.hermite.hermgauss(deg)
        t = t.reshape(-1,1)
        w = w.reshape(-1,1)
        if t_scale is not None:
            t *= t_scale
        if w_scale is not None:
            w *= w_scale
        self.t = torch.tensor(t, device=config.device, dtype=config.dtype)  # Mx1
        self.w = torch.tensor(w, device=config.device, dtype=config.dtype)  # Mx1
        self.deg = deg

    def __call__(self, mu, var, F):
        return F(mu + var.sqrt().mm(self.t.T)).mm(self.w)  # Nx1
