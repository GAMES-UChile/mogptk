import sys
import torch
import numpy as np
from IPython.display import display, HTML
from . import Parameter, Mean, Kernel, GaussianLikelihood, config
from functools import reduce
import operator

def prod(iterable):
    return reduce(operator.mul, iterable, 1)

class CholeskyException(Exception):
    def __init__(self, message, K, model):
        self.message = message
        self.K = K
        self.model = model

    def __str__(self):
        return self.message

class Model:
    def __init__(self, kernel, X, y, mean=None, name=None):
        if not issubclass(type(kernel), Kernel):
            raise ValueError("kernel must derive from mogptk.gpr.Kernel")
        X, y = self._check_input(X, y)
        if mean is not None:
            if not issubclass(type(mean), Mean):
                raise ValueError("mean must derive from mogptk.gpr.Mean")
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
    
    def print_parameters(self, file=None):
        def param_range(lower, upper, trainable=True):
            if lower is not None:
                if prod(lower.shape) == 1:
                    lower = lower.item()
                else:
                    lower = lower.tolist()
            if upper is not None:
                if prod(upper.shape) == 1:
                    upper = upper.item()
                else:
                    upper = upper.tolist()

            if not trainable:
                return "fixed"
            if lower is None and upper is None:
                return "(-∞, ∞)"
            elif lower is None:
                return "(-∞, %s]" % upper
            elif upper is None:
                return "[%s, ∞)" % lower
            return "[%s, %s]" % (lower, upper)

        if file is None:
            try:
                get_ipython  # fails if we're not in a notebook
                table = '<table><tr><th style="text-align:left">Name</th><th>Range</th><th>Value</th></tr>'
                for name, p in zip(self._param_names, self._params):
                    table += '<tr><td style="text-align:left">%s</td><td>%s</td><td>%s</td></tr>' % (name, param_range(p.lower, p.upper, p.trainable), p.numpy())
                table += '</table>'
                display(HTML(table))
                return
            except Exception as e:
                pass

        vals = [["Name", "Range", "Value"]]
        for name, p in zip(self._param_names, self._params):
            vals.append([name, param_range(p.lower, p.upper, p.trainable), str(p.numpy())])

        nameWidth = max([len(val[0]) for val in vals])
        rangeWidth = max([len(val[1]) for val in vals])
        for val in vals:
            print("%-*s  %-*s  %s" % (nameWidth, val[0], rangeWidth, val[1], val[2]), file=file)

    def _cholesky(self, K):
        try:
            return torch.linalg.cholesky(K)
        except RuntimeError as e:
            print("ERROR:", e.args[0], file=sys.__stdout__)
            print("K =", K, file=sys.__stdout__)
            if K.isnan().any():
                print("Kernel matrix has NaNs!", file=sys.__stdout__)
            if K.isinf().any():
                print("Kernel matrix has infinities!", file=sys.__stdout__)
            print("Parameters:", file=sys.__stdout__)
            self.print_parameters(file=sys.__stdout__)
            raise CholeskyException(e.args[0], K, self)

    def log_marginal_likelihood(self):
        raise NotImplementedError()

    def log_prior(self):
        return sum([p.log_prior() for p in self._params])

    def loss(self):
        self.zero_grad()
        loss = -self.log_marginal_likelihood() - self.log_prior()
        loss.backward()
        return loss

    def K(self, X1, X2=None):
        with torch.no_grad():
            X1 = self._check_input(X1)  # MxD
            if X2 is not None:
                X2 = self._check_input(X2)  # MxD
            return self.kernel(X1,X2).cpu().numpy() # does cpu().numpy() detach? check memory usage

    def sample(self, Z, n=None):
        with torch.no_grad():
            S = n
            if n is None:
                S = 1

            mu, var = self.predict(Z, full=True, tensor=True)  # MxD and MxMxD
            u = torch.normal(
                    torch.zeros(Z.shape[0], S, device=config.device, dtype=config.dtype),
                    torch.tensor(1.0, device=config.device, dtype=config.dtype))  # MxS
            L = torch.linalg.cholesky(var + 1e-6*torch.ones(Z.shape[0]).diagflat())  # MxM
            samples = mu + L.mm(u)  # MxS
            if n is None:
                samples = samples.squeeze()
            return samples.cpu().numpy()

class GPR(Model):
    def __init__(self, kernel, X, y, variance=1.0, mean=None, name="GPR"):
        super(GPR, self).__init__(kernel, X, y, mean, name)

        self.eye = torch.eye(self.X.shape[0], device=config.device, dtype=config.dtype)
        self.log_marginal_likelihood_constant = 0.5*X.shape[0]*np.log(2.0*np.pi)

        self.variance = Parameter(variance, name="variance", lower=config.positive_minimum)
        self._register_parameters(self.variance)

    def log_marginal_likelihood(self):
        K = self.kernel(self.X) + self.variance()*self.eye  # NxN
        L = self._cholesky(K)  # NxN

        if self.mean is not None:
            y = self.y - self.mean(self.X).reshape(-1,1)  # Nx1
        else:
            y = self.y  # Nx1

        p = self.log_marginal_likelihood_constant
        p -= L.diagonal().log().sum() # 0.5 is taken inside the log: L is the square root
        p -= 0.5*y.T.mm(torch.cholesky_solve(y,L)).squeeze()
        return p#/self.X.shape[0]  # dividing by the number of data points normalizes the learning rate

    def predict(self, Xs, full=False, tensor=False):
        with torch.no_grad():
            Xs = self._check_input(Xs)  # MxD

            K = self.kernel(self.X) + self.variance()*self.eye  # NxN
            Ks = self.kernel(self.X,Xs)  # NxM

            L = self._cholesky(K)  # NxN
            v = torch.triangular_solve(Ks,L,upper=False)[0]  # NxM

            if self.mean is not None:
                y = self.y - self.mean(self.X).reshape(-1,1)  # Nx1
                mu = Ks.T.mm(torch.cholesky_solve(y,L))  # Mx1
                mu += self.mean(Xs).reshape(-1,1)  # Mx1
            else:
                mu = Ks.T.mm(torch.cholesky_solve(self.y,L))  # Mx1

            if full:
                Kss = self.kernel(Xs)  # MxM
                var = Kss - v.T.mm(v)  # MxM
            else:
                Kss_diag = self.kernel.K_diag(Xs)  # Mx1
                var = Kss_diag - v.T.square().sum(dim=1)  # Mx1
                var = var.reshape(-1,1)

            if tensor:
                return mu, var
            else:
                return mu.cpu().numpy(), var.cpu().numpy()

class Sparse(Model):
    # Using http://krasserm.github.io/2020/12/12/gaussian-processes-sparse/
    def __init__(self, kernel, X, y, Z, variance=1.0, jitter=1e-8,
                 mean=None, name="Sparse"):
        super(Sparse, self).__init__(kernel, X, y, mean, name)

        Z = self._check_input(Z)
        self.Z = Parameter(Z, name="induction_points")
        self._register_parameters(self.Z)

        self.jitter = jitter
        if variance < jitter:
            variance = jitter
        self.variance = Parameter(variance, name="variance", lower=config.positive_minimum)
        self._register_parameters(self.variance)

        self.eye = torch.eye(Z.shape[0], device=config.device, dtype=config.dtype)
        self.log_marginal_likelihood_constant = 0.5*X.shape[0]*np.log(2.0*np.pi)

    def elbo(self):
        if self.mean is not None:
            y = self.y - self.mean(self.X).reshape(-1,1)  # Nx1
        else:
            y = self.y  # Nx1

        Knn_diag = self.kernel.K_diag(self.X)  # Nx1
        Kmn = self.kernel(self.Z(),self.X)  # MxN
        Kmm = self.kernel(self.Z()) + self.jitter*self.eye  # MxM

        Lmm = self._cholesky(Kmm)  # MxM;  Lmm = Kmm^(1/2)
        v = torch.triangular_solve(Kmn,Lmm,upper=False)[0]  # MxN;  v = Kmm^(-1/2) . Kmn
        Q = v.mm(v.T)  # MxM;  Q = Kmm^(-1/2) . Kmn . Knm . Kmm^(-1/2)
        L = self._cholesky(Q/self.variance() + self.eye)  # MxM;  L = (Q/var + I)^(1/2)

        c = torch.triangular_solve(v.mm(y),L,upper=False)[0]/self.variance()  # Mx1;  c = L^(-1) . Kmm^(-1/2) . Kmn . y

        p = -self.log_marginal_likelihood_constant
        p -= L.diagonal().log().sum() # 0.5 is taken as the square root of L
        p -= 0.5*self.X.shape[0]*self.variance().log()
        p -= 0.5*y.T.mm(y).squeeze()/self.variance()
        p += 0.5*c.T.mm(c).squeeze()
        p -= 0.5*(Knn_diag.sum() - Q.trace())/self.variance() # trace
        return p

    def log_marginal_likelihood(self):
        # maximize the lower bound
        return self.elbo()

    def predict(self, Xs, full=False, tensor=False):
        with torch.no_grad():
            Xs = self._check_input(Xs)  # MxD
            if self.mean is not None:
                y = self.y - self.mean(self.X).reshape(-1,1)  # Nx1
            else:
                y = self.y  # Nx1

            Kms = self.kernel(self.Z(),Xs)  # MxS
            Kmn = self.kernel(self.Z(),self.X)  # MxN
            Kmm = self.kernel(self.Z()) + self.jitter*self.eye  # MxM

            Lmm = self._cholesky(Kmm)  # MxM;  Lmm = Kmm^(1/2)
            v = torch.triangular_solve(Kmn,Lmm,upper=False)[0]  # MxN;  v = Kmm^(-1/2) . Kmn
            Q = v.mm(v.T)  # MxM;  Q = Kmm^(-1/2) . Kmn . Knm . Kmm^(-1/2)
            L = self._cholesky(Q/self.variance() + self.eye)  # MxM;  L = (Q/var + I)^(1/2)

            a = torch.triangular_solve(Kms,Lmm,upper=False)[0]  # MxS;  Kmm^(-1/2) . Kms
            b = torch.triangular_solve(a,L,upper=False)[0]  # MxS;  L^(-1) . Kmm^(-1/2) . Kms
            c = torch.triangular_solve(v.mm(y),L,upper=False)[0]/self.variance()  # Mx1;  c = L^(-1) . Kmm^(-1/2) . Kmn . y

            mu = b.T.mm(c)  # Mx1
            if self.mean is not None:
                mu += self.mean(Xs).reshape(-1,1)  # Mx1

            if full:
                Kss = self.kernel(Xs)  # MxM
                var = Kss - a.T.mm(a) + b.T.mm(b)  # MxM
            else:
                Kss_diag = self.kernel.K_diag(Xs)  # Mx1
                var = Kss_diag - a.T.square().sum(dim=1) + b.T.square().sum(dim=1)  # Mx1
                var = var.reshape(-1,1)

            if tensor:
                return mu, var
            else:
                return mu.cpu().numpy(), var.cpu().numpy()

class SparseVariational(Model):
    # Using http://krasserm.github.io/2020/12/12/gaussian-processes-sparse/
    def __init__(self, kernel, X, y, Z, likelihood=Gaussian, jitter=1e-8,
                 mean=None, name="SparseVariational"):
        super(SparseVariational, self).__init__(kernel, X, y, mean, name)

        Z = self._check_input(Z)

        self.jitter = jitter
        if variance < jitter:
            variance = jitter

        self.eye = torch.eye(Z.shape[0], device=config.device, dtype=config.dtype)
        self.log_marginal_likelihood_constant = 0.5*X.shape[0]*np.log(2.0*np.pi)

        self.Z = Parameter(Z, name="induction_points")
        self.q_mu = Parameter(torch.zeros(len(Z),1), name="q_mu")
        self.q_var = Parameter(torch.eye(len(Z)), name="q_var", lower=config.positive_minimum)

        self._register_parameters(self.Z)
        self._register_parameters(self.q_mu)
        self._register_parameters(self.q_var)

    def kl_gaussian(self, q_mu, q_var, p_mu, p_var):
        Lq = self._cholesky(q_var) # NxN
        Lp = self._cholesky(p_var) # NxN
        v = torch.triangular_solve(p_mu-q_mu,Lp,upper=False)[0]  # Nx1

        kl = 0.5*p_mu.shape[0]
        kl += 0.5*v.T.mm(v)
        kl += Lp.diagonal().log().sum()
        kl -= Lq.diagonal().log().sum()
        kl += 0.5*torch.cholesky_solve(p_var,Lq).diagonal().sum()
        return kl

    def elbo(self):
        p_mu = torch.zeros(len(self.Z),1)
        p_var = selk.kernel(self.Z) + self.jitter*self.eye
        return p - self.kl_gaussian(self.q_mu, self.q_var, p_mu, p_var)

    def log_marginal_likelihood(self):
        # maximize the lower bound
        return self.elbo()

    def predict(self, Xs, full=False, tensor=False):
        with torch.no_grad():
            Xs = self._check_input(Xs)  # MxD
            if self.mean is not None:
                y = self.y - self.mean(self.X).reshape(-1,1)  # Nx1
            else:
                y = self.y  # Nx1

            Kms = self.kernel(self.Z(),Xs)  # MxS
            Kmn = self.kernel(self.Z(),self.X)  # MxN
            Kmm = self.kernel(self.Z()) + self.jitter*self.eye  # MxM

            Lmm = self._cholesky(Kmm)  # MxM;  Lmm = Kmm^(1/2)
            v = torch.triangular_solve(Kmn,Lmm,upper=False)[0]  # MxN;  v = Kmm^(-1/2) . Kmn
            Q = v.mm(v.T)  # MxM;  Q = Kmm^(-1/2) . Kmn . Knm . Kmm^(-1/2)
            L = self._cholesky(Q/self.variance() + self.eye)  # MxM;  L = (Q/var + I)^(1/2)

            a = torch.triangular_solve(Kms,Lmm,upper=False)[0]  # MxS;  Kmm^(-1/2) . Kms
            b = torch.triangular_solve(a,L,upper=False)[0]  # MxS;  L^(-1) . Kmm^(-1/2) . Kms
            c = torch.triangular_solve(v.mm(y),L,upper=False)[0]/self.variance()  # Mx1;  c = L^(-1) . Kmm^(-1/2) . Kmn . y

            mu = b.T.mm(c)  # Mx1
            if self.mean is not None:
                mu += self.mean(Xs).reshape(-1,1)  # Mx1

            if full:
                Kss = self.kernel(Xs)  # MxM
                var = Kss - a.T.mm(a) + b.T.mm(b)  # MxM
            else:
                Kss_diag = self.kernel.K_diag(Xs)  # Mx1
                var = Kss_diag - a.T.square().sum(dim=1) + b.T.square().sum(dim=1)  # Mx1
                var = var.reshape(-1,1)

            if tensor:
                return mu, var
            else:
                return mu.cpu().numpy(), var.cpu().numpy()
