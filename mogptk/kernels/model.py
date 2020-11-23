import torch
import numpy as np
from IPython.display import display, HTML
from . import Parameter, Mean, Kernel, config

class CholeskyException(Exception):
    pass

class Model:
    def __init__(self, kernel, X, y, mean=None, name=None):
        if not issubclass(type(kernel), Kernel):
            raise ValueError("kernel must derive from mogptk.kernels.Kernel")
        X, y = self._check_input(X, y)
        if mean is not None:
            if not issubclass(type(mean), Mean):
                raise ValueError("mean must derive from mogptk.kernels.Mean")
            mu = mean(X).reshape(-1,1)
            if mu.shape != y.shape:
                raise ValueError("mean and y data must match shapes: %s != %s" % (mu.shape, y.shape))

        self.kernel = kernel
        self.X = X
        self.y = y
        self.mean = mean
        self.name = name
        self.input_dims = X.shape[1]

        self._params = []
        self._param_names = []
        self._register_parameters(kernel)
        if mean is not None and issubclass(type(mean), Mean):
            self._register_parameters(mean)

    def __setattr__(self, name, val):
        if hasattr(self, name) and isinstance(getattr(self, name), Parameter):
            raise AttributeError("parameter is read-only, use Parameter.assign()")
        if isinstance(val, Parameter) and val.name is None:
            val.name = name
        super(Model,self).__setattr__(name, val)        

    def _check_input(self, X, y=None):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, device=config.device, dtype=config.dtype)
        else:
            X = X.to(config.device, config.dtype)
        if len(X.shape) == 1:
            X = X.reshape(-1,1)
        if len(X.shape) != 2:
            raise ValueError("X must have dimensions (data_points,input_dims) with input_dims optional")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must not be empty")

        if y is not None:
            if not isinstance(y, torch.Tensor):
                y = torch.tensor(y, device=config.device, dtype=config.dtype)
            else:
                y = y.to(config.device, config.dtype)
            if len(y.shape) == 1:
                y = y.reshape(-1,1)
            if len(y.shape) != 2 or y.shape[1] != 1:
                raise ValueError("y must have one dimension (data_points,)")
            if X.shape[0] != y.shape[0]:
                raise ValueError("number of data points for X and y must match")
            return X, y
        else:
            # X is for prediction
            if X.shape[1] != self.input_dims:
                raise ValueError("X must have %s input dimensions" % self.input_dims)
            return X

    def _register_parameters(self, obj, name=None):
        if isinstance(obj, Parameter):
            if obj.name is not None:
                if name is None:
                    name = obj.name
                else:
                    name += "." + obj.name
            elif name is None:
                name = ""
            self._params.append(obj)
            self._param_names.append(name)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                self._register_parameters(v, (name if name is not None else "")+"["+str(i)+"]")
        elif issubclass(type(obj), Kernel) or issubclass(type(obj), Mean):
            for v in obj.__dict__.values():
                self._register_parameters(v, (name+"." if name is not None else "")+obj.name)

    def zero_grad(self):
        for p in self._params:
            p = p.unconstrained
            if p.grad is not None:
                if p.grad.grad_fn is not None:
                    p.grad.detach_()
                else:
                    p.grad.requires_grad_(False)
                p.grad.zero_()

    def parameters(self):
        for p in self._params:
            if p.trainable:
                yield p.unconstrained
    
    def print_parameters(self):
        def param_range(lower, upper, trainable=True):
            if not trainable:
                return "fixed"
            if lower is None and upper is None:
                return "(-∞,∞)"
            elif lower is None:
                return "(-∞,%s]" % upper.tolist()
            elif upper is None:
                return "[%s,∞)" % lower.tolist()
            return "[%s,%s]" % (lower.tolist(),upper.tolist())

        try:
            get_ipython  # fails if we're not in a notebook
            table = '<table><tr><th style="text-align:left">Name</th><th>Range</th><th>Value</th></tr>'
            for name, p in zip(self._param_names, self._params):
                table += '<tr><td style="text-align:left">%s</td><td>%s</td><td>%s</td></tr>' % (name, param_range(p.lower, p.upper, p.trainable), p.constrained.detach().numpy())
            table += '</table>'
            display(HTML(table))
        except Exception as e:
            vals = [["Name", "Range", "Value"]]
            for name, p in zip(self._param_names, self._params):
                vals.append([name, param_range(p.lower, p.upper, p.trainable), str(p.constrained.detach().numy())])

            nameWidth = max([len(val[0]) for val in vals])
            rangeWidth = max([len(val[1]) for val in vals])
            for val in vals:
                print("%-*s  %-*s  %s" % (nameWidth, val[0], rangeWidth, val[1], val[2]))

    def _cholesky(self, K):
        try:
            return torch.cholesky(K)
        except RuntimeError as e:
            print()
            print("ERROR:", e.args[0])
            print("K =", K)
            if K.isnan().any():
                print("Kernel matrix has NaNs!")
            if K.isinf().any():
                print("Kernel matrix has infinities!")
            print("Parameters:")
            self.print_parameters()
            raise CholeskyException

    def log_marginal_likelihood(self):
        raise NotImplementedError()

    def log_prior(self):
        return sum([p.log_prior() for p in self._params])

    def loss(self):
        self.zero_grad()
        loss = -self.log_marginal_likelihood() - self.log_prior()
        loss.backward()
        return loss

    def K(self, Z):
        with torch.no_grad():
            Z = self._check_input(Z)  # MxD
            return self.kernel(Z).detach().numpy()

    def sample(self, Z, n=None):
        with torch.no_grad():
            S = n
            if n is None:
                S = 1

            mu, var = self.predict(Z, full_var=True)  # MxD and MxMxD
            u = torch.normal(torch.zeros(Z.shape[0], S, device=config.device, dtype=config.dtype), torch.tensor(1.0, device=config.device, dtype=config.dtype))  # MxS
            L = torch.cholesky(var + 1e-6*torch.ones(Z.shape[0]).diagflat())  # MxM
            samples = mu + L.mm(u)  # MxS
            if num is None:
                samples = samples.squeeze()
            return samples.detach().numpy()

class GPR(Model):
    def __init__(self, kernel, X, y, noise=1.0, mean=None, name="GPR"):
        super(GPR, self).__init__(kernel, X, y, mean, name)

        self.log_marginal_likelihood_constant = 0.5*X.shape[0]*np.log(2.0*np.pi)

        self.noise = Parameter(noise, name="noise", lower=config.positive_minimum)
        self._register_parameters(self.noise)

    def log_marginal_likelihood(self):
        K = self.kernel(self.X) + self.noise()*torch.eye(self.X.shape[0], device=config.device, dtype=config.dtype)  # NxN
        L = self._cholesky(K)  # NxN

        if self.mean is not None:
            y = self.y - self.mean(self.X).reshape(-1,1)  # Nx1
        else:
            y = self.y  # Nx1

        p = -0.5*y.T.mm(torch.cholesky_solve(y,L)).squeeze()
        p -= L.diagonal().log().sum()
        p -= self.log_marginal_likelihood_constant
        return p#/self.X.shape[0]  # dividing by the number of data points normalizes the learning rate

    def predict(self, Z):
        with torch.no_grad():
            Z = self._check_input(Z)  # MxD

            K = self.kernel(self.X) + self.noise()*torch.eye(self.X.shape[0], device=config.device, dtype=config.dtype)  # NxN
            Ks = self.kernel(self.X,Z)  # NxM
            Kss = self.kernel(Z) + self.noise()*torch.eye(Z.shape[0], device=config.device, dtype=config.dtype)  # MxM

            L = self._cholesky(K)  # NxN
            v = torch.triangular_solve(Ks,L,upper=False)[0]  # NxM

            if self.mean is not None:
                y = self.y - self.mean(self.X).reshape(-1,1)  # Nx1
                mu = Ks.T.mm(torch.cholesky_solve(y,L))  # Mx1
                mu += self.mean(Z).reshape(-1,1)         # Mx1
            else:
                mu = Ks.T.mm(torch.cholesky_solve(self.y,L))  # Mx1

            var = Kss - v.T.mm(v)  # MxM
            var = var.diag().reshape(-1,1)  # Mx1
            return mu.detach().numpy(), var.detach().numpy()
