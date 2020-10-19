import torch
import numpy as np
from IPython.display import display, HTML
from . import Parameter, Kernel, device, dtype, positive_minimum

class Mean:
    def __init__(self, name=None):
        if name is None:
            name = self.__class__.__name__
            if name.endswith('Mean'):
                name = name[:-4]
        self.name = name

    def __call__(self, X):
        raise NotImplementedError()

    def __setattr__(self, name, val):
        if hasattr(self, name) and isinstance(getattr(self, name), Parameter):
            raise AttributeError("parameter is read-only, use Parameter.assign()")
        if isinstance(val, Parameter):
            val.parent = self
            if val.name is None:
                val.name = name
        super(Mean,self).__setattr__(name, val)        

class Model:
    def __init__(self, name=None):
        self.name = name
        self.mean = None
        self._params = []

    def _check_input(self, X, y=None):
        if len(X.shape) != 2:
            raise ValueError("X should have dimensions (data_points,input_dims) with input_dims optional")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must not be empty")

        if y is not None:
            if len(y.shape) != 2 or y.shape[1] != 1:
                raise ValueError("y should have one dimension (data_points,)")
            if X.shape[0] != y.shape[0]:
                raise ValueError("number of data points for X and y must match")

    def _register_parameters(self, obj):
        if isinstance(obj, Parameter):
            self._params.append(obj)
        elif isinstance(obj, list):
            for v in obj:
                self._register_parameters(v)
        elif issubclass(type(obj), Kernel) or issubclass(type(obj), Mean):
            for v in obj.__dict__.values():
                self._register_parameters(v)

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
        def param_range(lower, upper):
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
            for p in self._params:
                name = p.name
                if name is None:
                    name = ""
                elif p.parent is not None:
                    name = p.parent.name + "." + name
                table += '<tr><td style="text-align:left">%s</td><td>%s</td><td>%s</td></tr>' % (name, param_range(p.lower, p.upper), p.constrained.detach().numpy())
            table += '</table>'
            display(HTML(table))
        except Exception as e:
            vals = [["name", "range", "value"]]
            for p in self._params:
                name = p.name
                if name is None:
                    name = ""
                elif p.parent is not None:
                    name = p.parent.name + "." + name
                vals.append([name, str(p.transform), str(p.constrained.detach().numy())])

            nameWidth = max([len(val[0]) for val in vals])
            rangeWidth = max([len(val[1]) for val in vals])
            for val in vals:
                print("%-*s  %-*s  %s" % (nameWidth, val[0], rangeWidth, val[1], val[2]))

    def log_marginal_likelihood(self):
        raise NotImplementedError()

    def log_prior(self):
        return sum([p.log_prior() for p in self._params])

    def loss(self):
        self.zero_grad()
        loss = -self.log_marginal_likelihood() - self.log_prior()
        loss.backward()
        return loss

class GPR(Model):
    def __init__(self, kernel, X, y, variance=1.0, name="GPR", mean=None):
        super(GPR, self).__init__(name)

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, device=device, dtype=dtype)
        else:
            X = X.to(device, dtype)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, device=device, dtype=dtype)
        else:
            y = y.to(device, dtype)

        if len(X.shape) == 1:
            X = X.reshape(-1,1)
        if len(y.shape) == 1:
            y = y.reshape(-1,1)
        self._check_input(X, y)

        if mean is not None:
            mu = mean(X).reshape(-1,1)
            if mu.shape != y.shape:
                raise ValueError("mean and y must match shapes: %s != %s" % (mu.shape, y.shape))

        self.kernel = kernel
        self.X = X
        self.y = y
        self.variance = Parameter(variance, name="variance", lower=1e-6)
        self.mean = mean
        
        self.N = X.shape[0]
        self.input_dims = X.shape[1]
        self.log_marginal_likelihood_constant = 0.5*self.N*np.log(2.0*np.pi)

        self._register_parameters(self.variance)
        if mean is not None and issubclass(type(mean), Mean):
            self._register_parameters(mean)
        self._register_parameters(kernel)

    def log_marginal_likelihood(self):
        K = self.kernel(self.X) + self.variance()*torch.eye(self.N)
        try:
            L = torch.cholesky(K)
        except RuntimeError:
            print("K =", K)
            if K.isnan().any():
                print("Kernel matrix has NaNs!")
            if K.isinf().any():
                print("Kernel matrix has infinities!")
            print("Parameters:")
            self.print_parameters()
            raise

        if self.mean is not None:
            y = self.y - self.mean(X).reshape(-1,1)
        else:
            y = self.y

        p = -0.5*y.T.mm(torch.cholesky_solve(y,L)).squeeze()
        p -= L.diagonal().log().sum()
        p -= self.log_marginal_likelihood_constant
        return p

    def predict(self, Z):
        if not isinstance(Z, torch.Tensor):
            Z = torch.tensor(Z, device=device, dtype=dtype)
        else:
            Z = Z.to(device, dtype)

        if len(Z.shape) == 1:
            Z = Z.reshape(-1,1)
        self._check_input(Z)
        if Z.shape[1] != self.input_dims:
            raise ValueError("X must have %s input dimensions" % self.input_dims)

        with torch.no_grad():
            K = self.kernel(self.X) + self.variance()*torch.eye(self.N)
            Ks = self.kernel(self.X,Z)
            Kss = self.kernel(Z)

            L = torch.cholesky(K)
            v = torch.triangular_solve(Ks,L,upper=False)[0]

            if self.mean is not None:
                y = self.y - self.mean(X).reshape(-1,1)
                mu = Ks.T.mm(torch.cholesky_solve(y,L))
                mu += self.mean(Z).reshape(-1,1)
            else:
                mu = Ks.T.mm(torch.cholesky_solve(self.y,L))

            var = Kss - v.T.mm(v)
            var = var.diag().reshape(-1,1)
        return mu.detach().numpy(), var.detach().numpy()
