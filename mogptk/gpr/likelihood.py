import torch
import numpy as np
from . import config, Parameter

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

class Likelihood:
    def __init__(self, name="Likelihood", quadratures=20):
        self.name = name
        self.quadrature = GaussHermiteQuadrature(deg=quadratures, t_scale=np.sqrt(2), w_scale=1.0/np.sqrt(np.pi))
        self.output_dims = 1

    def log_prob(self, y, f, c=None):
        # log(p(y|f)), where p(y|f) is our likelihood
        # y: Nx1
        # f: NxM
        raise NotImplementedError()

    def variational_expectation(self, y, mu, var, c=None):
        # ∫ log(p(y|f)) q(f) df, where q(f) ~ N(mu,var) and p(y|f) is our likelihood
        # y,mu,var: Nx1
        q = self.quadrature(mu, var, lambda f: self.log_prob(y,f,c=c))  # Nx1
        return q.sum()  # sum over N

    def predictive_y(self, f, c=None):
        # f: NxM
        raise NotImplementedError()

    def predictive_yy(self, f, c=None):
        # f: NxM
        raise NotImplementedError()

    def predict(self, mu, var, full=False, c=None):
        # ∫∫ y p(y|f) q(f) df dy,  ∫∫ y^2 p(y|f) q(f) df dy - (∫∫ y p(y|f) q(f) df dy)^2
        # where q(f) ~ N(mu,var) and p(y|f) is our likelihood
        # mu,var: Nx1
        Ey = self.quadrature(mu, var, lambda f: self.predictive_y(f,c=c))
        Eyy = self.quadrature(mu, var, lambda f: self.predictive_yy(f,c=c))
        return Ey, Eyy-Ey**2 

class MultiOutputLikelihood(Likelihood):
    def __init__(self, *likelihoods, name="MultiOutputLikelihood", quadratures=20):
        super().__init__(name=name, quadratures=quadratures)

        if isinstance(likelihoods, tuple):
            if len(likelihoods) == 1 and isinstance(likelihoods[0], list):
                likelihoods = likelihoods[0]
            else:
                likelihoods = list(likelihoods)
        elif not isinstance(likelihoods, list):
            likelihoods = [likelihoods]
        if len(likelihoods) == 0:
            raise ValueError("must pass at least one likelihood")
        for i, likelihood in enumerate(likelihoods):
            if not issubclass(type(likelihood), Likelihood):
                raise ValueError("must pass likelihoods")
            elif isinstance(likelihood, MultiOutputLikelihood):
                raise ValueError("can not nest MultiOutputLikelihoods")

        self.output_dims = len(likelihoods)
        self.likelihoods = likelihoods

    def _indices(self, c):
        m = [c==j for j in range(self.output_dims)]
        r = [torch.nonzero(m[j], as_tuple=False).reshape(-1) for j in range(self.output_dims)]  # as_tuple avoids warning
        return r

    def log_prob(self, y, f, c=None):
        # y: Nx1
        # f: NxM
        if self.output_dims == 1:
            return self.likelihoods[0].log_prob(y,f)

        r = self._indices(c)
        res = torch.empty(f.shape, device=config.device, dtype=config.dtype)
        for i in range(self.output_dims):
            res[r[i],:] = self.likelihoods[i].log_prob(y[r[i],:], f[r[i],:])
        return res  # NxM

    def variational_expectation(self, y, mu, var, c=None):
        # y,mu,var: Nx1
        if self.output_dims == 1:
            return self.likelihoods[0].variational_expectation(y,mu,var)

        q = torch.tensor(0.0, dtype=config.dtype, device=config.device)
        r = self._indices(c)
        for i in range(self.output_dims):
            q += self.likelihoods[i].variational_expectation(y[r[i],:], mu[r[i],:], var[r[i],:]).sum()  # sum over N
        return q

    def predictive_y(self, f, c=None):
        # f: NxM
        if self.output_dims == 1:
            return self.likelihoods[0].predictive_y(f)

        r = self._indices(c)
        res = torch.empty(f.shape, device=config.device, dtype=config.dtype)
        for i in range(self.output_dims):
            res[r[i],:] = self.likelihoods[i].predictive_y(f[r[i],:])
        return res  # NxM

    def predictive_yy(self, f, c=None):
        # f: NxM
        if self.output_dims == 1:
            return self.likelihoods[0].predictive_yy2(f)

        r = self._indices(c)
        res = torch.empty(f.shape, device=config.device, dtype=config.dtype)
        for i in range(self.output_dims):
            res[r[i],:] = self.likelihoods[i].predictive_yy(f[r[i],:])
        return res  # NxM

    # TODO: predict is not possible?
    #def predict(self, mu, var, full=False, c=None):
    #    # mu: Nx1
    #    # var: Nx1 or NxN
    #    if self.output_dims == 1:
    #        return self.likelihoods[0].predict(mu,var,full=full)

    #    r = self._indices(c)
    #    Ey = torch.empty(mu.shape, device=config.device, dtype=config.dtype)
    #    Eyy = torch.empty(var.shape, device=config.device, dtype=config.dtype)
    #    if full:
    #        for i in range(self.output_dims):
    #            r1 = r[i].reshape(-1,1)
    #            r2 = r[i].reshape(1,-1)
    #            Ey[r[i],:], Eyy[r1,r2] = self.likelihoods[i].predict(mu[r[i],:], var[r1,r2], full=True)
    #    else:
    #        for i in range(self.output_dims):
    #            Ey[r[i],:], Eyy[r[i],:] = self.likelihoods[i].predict(mu[r[i],:], var[r[i],:], full=False)
    #    return Ey, Eyy-Ey.square()

class GaussianLikelihood(Likelihood):
    def __init__(self, variance=1.0, name="Gaussian", quadratures=20):
        super().__init__(name, quadratures)

        self.variance = Parameter(variance, name="variance", lower=config.positive_minimum)

    def log_prob(self, y, f, c=None):
        # y: Nx1
        # f: NxM
        p = -0.5 * (np.log(2.0 * np.pi) + self.variance().log() + (y-f).square()/self.variance())
        return p  # NxM

    def variational_expectation(self, y, mu, var, c=None):
        # y,mu,var: Nx1
        p = -((y-mu).square() + var) / self.variance()
        p -= np.log(2.0 * np.pi)
        p -= self.variance().log()
        return 0.5*p.sum()  # sum over N

    def predictive_y(self, f, c=None):
        return f

    def predictive_yy(self, f, c=None):
        return f.square() + self.variance()

    def predict(self, mu, var, c=None, full=False):
        if full:
            return mu, var + self.variance()*torch.eye(var.shape[0])
        else:
            return mu, var + self.variance()

class StudentTLikelihood(Likelihood):
    def __init__(self, dof=3, scale=1.0, name="StudentT", quadratures=20):
        super().__init__(name, quadratures)

        self.dof = torch.tensor(dof, device=config.device, dtype=config.dtype)
        self.scale = Parameter(scale, name="scale", lower=config.positive_minimum)

    def log_prob(self, y, f, c=None):
        # y: Nx1
        # f: NxM
        p = -0.5 * (self.dof+1.0)*torch.log(1.0 + ((y-f)/self.scale()).square()/self.dof)
        p += torch.lgamma((self.dof+1.0)/2.0)
        p -= torch.lgamma(self.dof/2.0)
        p -= 0.5 * torch.log(self.dof*np.pi*self.scale()**2)
        return p  # NxM

    def predictive_y(self, f, c=None):
        if self.dof <= 1.0:
            return torch.full(f.shape, np.nan, device=config.device, dtype=config.dtype)
        return f

    def predictive_yy(self, f, c=None):
        if self.dof <= 2.0:
            return torch.full(f.shape, np.nan, device=config.device, dtype=config.dtype)
        return f.square() + self.scale()**2 * self.dof/(self.dof-2.0)

class LaplaceLikelihood(Likelihood):
    def __init__(self, scale=1.0, name="Laplace", quadratures=20):
        super().__init__(name, quadratures)

        self.scale = Parameter(scale, name="scale", lower=config.positive_minimum)

    def log_prob(self, y, f, c=None):
        # y: Nx1
        # f: NxM
        p = -torch.log(2.0*self.scale()) - (y-f).abs()/self.scale()
        return p  # NxM

    def predictive_y(self, f, c=None):
        return f

    def predictive_yy(self, f, c=None):
        return f.square() + 2.0*self.scale()**2

def inv_probit(x):
    jitter = 1e-3
    return 0.5*(1.0+torch.erf(x/np.sqrt(2.0))) * (1.0-2.0*jitter) + jitter

def logistic(x):
    return 1.0/(1.0+torch.exp(-x))

class BernoulliLikelihood(Likelihood):
    def __init__(self, scale=1.0, link=inv_probit, name="Bernoulli", quadratures=20):
        super().__init__(name, quadratures)

        self.link = link

    def log_prob(self, y, f, c=None):
        # y: Nx1
        # f: NxM
        p = self.link(f)
        return torch.log(torch.where(0.5 <= y, p, 1.0-p))  # NxM

    def predictive_y(self, f, c=None):
        return self.link(f)

    def predictive_yy(self, f, c=None):
        return self.link(f)

    def predict(self, mu, var, c=None, full=False):
        if self.link == inv_probit:
            p = self.link(mu / torch.sqrt(1.0 + var))
            if full:
                return p.diagonal().reshape(-1,1), p-p.square() # TODO: correct?
            return p, p-p.square()
        else:
            return super().predict(mu, var, c=c, full=full)

# TODO: implement log_prob: Beta, Softmax
# TODO: implement log_prob and var_exp: Gamma, Laplace, Poisson
