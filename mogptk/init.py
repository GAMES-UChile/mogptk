import torch
import numpy as np
from . import gpr

def BNSE(x, y, y_err=None, max_freq=None, n=1000, iters=100):
    """
    Bayesian non-parametric spectral estimation [1] is a method for estimating the power spectral density of a signal that uses a Gaussian process with a spectral mixture kernel to learn the spectral representation of the signal. The resulting power spectral density is distributed as a generalized Chi-Squared distributions.

    Args:
        x (numpy.ndarray): Input data of shape (data_points,).
        y (numpy.ndarray): Output data of shape (data_points,).
        y_err (numpy.ndarray): Output std.dev. data of shape (data_points,).
        max_freq (float): Maximum frequency of the power spectral density. If not given the Nyquist frequency is estimated and used instead.
        n (int): Number of points in frequency space to sample the power spectral density.
        iters (int): Number of iterations used to train the Gaussian process.

    Returns:
        numpy.ndarray: Frequencies of shape (n,).
        numpy.ndarray: Power spectral density mean of shape (n,).
        numpy.ndarray: Power spectral density variance of shape (n,).

    [1] F. Tobar, Bayesian nonparametric spectral estimation, Advances in Neural Information Processing Systems, 2018
    """
    x -= np.median(x)
    x_range = np.max(x)-np.min(x)
    x_dist = x_range/len(x)
    if max_freq is None:
        max_freq = 0.5/x_dist

    x = torch.tensor(x, device=gpr.config.device, dtype=gpr.config.dtype)
    if x.ndim == 0:
        x = x.reshape(1,1)
    elif x.ndim == 1:
        x = x.reshape(-1,1)
    y = torch.tensor(y, device=gpr.config.device, dtype=gpr.config.dtype).reshape(-1,1)

    kernel = gpr.SpectralKernel()
    model = gpr.Exact(kernel, x, y, data_variance=y_err**2 if y_err is not None else None)

    # initialize parameters
    magnitude = y.var()
    mean = 0.01
    variance = 0.25 / np.pi**2 / x_dist**2
    noise = y.std()/10.0
    model.kernel.magnitude.assign(magnitude)
    model.kernel.mean.assign(mean, upper=max_freq)
    model.kernel.variance.assign(variance)
    model.likelihood.scale.assign(noise)

    # train model
    optimizer = torch.optim.Adam(model.parameters(), lr=2.0)
    for i in range(iters):
        optimizer.step(model.loss)

    alpha = float(0.5/x_range**2)
    w = torch.linspace(0.0, max_freq, n, device=gpr.config.device, dtype=gpr.config.dtype).reshape(-1,1)

    def kernel_ff(f1, f2, magnitude, mean, variance, alpha):
        # f1,f2: MxD,  mean,variance: D
        mean = mean.reshape(1,1,-1)
        variance = variance.reshape(1,1,-1)
        gamma = 2.0*np.pi**2*variance
        const = 0.5 * np.pi * magnitude / torch.sqrt(alpha**2 + 2.0*alpha*gamma.prod())
        exp1 = -0.5 * np.pi**2 / alpha * gpr.Kernel.squared_distance(f1,f2)  # MxMxD
        exp2a = -2.0 * np.pi**2 / (alpha+2.0*gamma) * (gpr.Kernel.average(f1,f2)-mean)**2  # MxMxD
        exp2b = -2.0 * np.pi**2 / (alpha+2.0*gamma) * (gpr.Kernel.average(f1,f2)+mean)**2  # MxMxD
        return const * (torch.exp(exp1+exp2a) + torch.exp(exp1+exp2b)).sum(dim=2)

    def kernel_tf(t, f, magnitude, mean, variance, alpha):
        # t: NxD,  f: MxD,  mean,variance: D
        mean = mean.reshape(1,-1)
        variance = variance.reshape(1,-1)
        gamma = 2.0*np.pi**2*variance
        Lq_inv = np.pi**2 * (1.0/alpha + 1.0/gamma)  # 1xD
        Lq_inv = 1.0/Lq_inv  # this line must be kept, is this wrong in the paper?

        const = torch.sqrt(np.pi/(alpha+gamma.prod()))  # 1
        exp1 = -np.pi**2 * torch.tensordot(t**2, Lq_inv.T, dims=1)  # Nx1
        exp2a = -torch.tensordot(np.pi**2/(alpha+gamma), (f-mean).T**2, dims=1)  # 1xM
        exp2b = -torch.tensordot(np.pi**2/(alpha+gamma), (f+mean).T**2, dims=1)  # 1xM
        exp3a = -2.0*np.pi * torch.tensordot(t.mm(Lq_inv), np.pi**2 * (f/alpha + mean/gamma).T, dims=1)  # NxM
        exp3b = -2.0*np.pi * torch.tensordot(t.mm(Lq_inv), np.pi**2 * (f/alpha - mean/gamma).T, dims=1)  # NxM

        a = 0.5 * magnitude * const * torch.exp(exp1)
        real = torch.exp(exp2a)*torch.cos(exp3a) + torch.exp(exp2b)*torch.cos(exp3b)
        imag = torch.exp(exp2a)*torch.sin(exp3a) + torch.exp(exp2b)*torch.sin(exp3b)
        return a * real, a * imag

    with torch.no_grad():
        Ktt = kernel(x)
        Ktt += model.likelihood.scale().square() * torch.eye(x.shape[0], device=gpr.config.device, dtype=gpr.config.dtype)
        Ltt = model._cholesky(Ktt, add_jitter=True)

        Kff = kernel_ff(w, w, kernel.magnitude(), kernel.mean(), kernel.variance(), alpha)
        Pff = kernel_ff(w, -w, kernel.magnitude(), kernel.mean(), kernel.variance(), alpha)
        Kff_real = 0.5 * (Kff + Pff)
        Kff_imag = 0.5 * (Kff - Pff)

        Ktf_real, Ktf_imag = kernel_tf(x, w, kernel.magnitude(), kernel.mean(), kernel.variance(), alpha)

        a = torch.cholesky_solve(y,Ltt)
        b = torch.linalg.solve_triangular(Ltt,Ktf_real,upper=False)
        c = torch.linalg.solve_triangular(Ltt,Ktf_imag,upper=False)

        mu_real = Ktf_real.T.mm(a)
        mu_imag = Ktf_imag.T.mm(a)
        var_real = Kff_real - b.T.mm(b)
        var_imag = Kff_imag - c.T.mm(c)

        # The PSD equals N(mu_real,var_real)^2 + N(mu_imag,var_imag)^2, which is a generalized Chi-Squared distribution
        var_real = var_real.diagonal().reshape(-1,1)
        var_imag = var_imag.diagonal().reshape(-1,1)
        mu = mu_real**2 + mu_imag**2 + var_real + var_imag
        var = 2.0*var_real**2 + 2.0*var_imag**2 + 4.0*var_real*mu_real**2 + 4.0*var_imag*mu_imag**2

        w = w.cpu().numpy().reshape(-1)
        mu = mu.cpu().numpy().reshape(-1)
        var = var.cpu().numpy().reshape(-1)
    return w, mu, var
