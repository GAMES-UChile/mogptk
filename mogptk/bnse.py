import torch
import numpy as np
from . import gpr

def BNSE(x, y, max_freq=None, n=1000, iters=500, **params):
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
    model = gpr.Exact(kernel, x, y)

    # initialize parameters
    sigma = y.std()
    mean = 0.01
    variance = 0.25 / np.pi**2 / x_dist**2
    noise = y.std()/10.0
    model.kernel.sigma.assign(sigma)
    model.kernel.mean.assign(mean, upper=max_freq)
    model.kernel.variance.assign(variance)
    model.likelihood.scale.assign(noise)

    # train model
    optimizer = torch.optim.LBFGS(model.parameters(), max_iter=iters)
    loss = optimizer.step(model.loss)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001, iters=500)
    #for i in range(iters):
    #    loss = optimizer.step(model.loss)

    alpha = 0.5/x_range**2
    w = torch.linspace(0.0, max_freq, n, device=gpr.config.device, dtype=gpr.config.dtype).reshape(-1,1)

    def kernel_ff(f1, f2, sigma, mean, variance, alpha):
        # f1,f2: MxD,  mean,variance: D
        const = 0.5 * np.pi * sigma**2 / torch.sqrt(alpha**2 + 4.0*np.pi**2*alpha*variance.prod())
        mean = mean.reshape(1,1,-1)
        variance = variance.reshape(1,1,-1)
        exp1 = -0.5 * np.pi**2 / alpha * gpr.Kernel.squared_distance(f1,f2)  # MxMxD

        exp2a = -2.0 * np.pi**2 / (alpha+4.0*np.pi**2*variance) * (gpr.Kernel.average(f1,f2)-mean)**2  # MxMxD
        exp2b = -2.0 * np.pi**2 / (alpha+4.0*np.pi**2*variance) * (gpr.Kernel.average(f1,f2)+mean)**2  # MxMxD
        return const * (torch.exp(exp1+exp2a) + torch.exp(exp1+exp2b)).sum(dim=2)

    def kernel_tf(t, f, sigma, mean, variance, alpha):
        # t: NxD,  f: MxD,  mean,variance: D
        mean = mean.reshape(1,-1)
        variance = variance.reshape(1,-1)
        gamma = 2.0*np.pi**2*variance
        Lq_inv = np.pi**2 * (1.0/alpha + 1.0/gamma)  # 1xD
        Lq_inv = 1.0/Lq_inv  # TODO: this line must be kept, is this wrong in the paper?

        const = torch.sqrt(np.pi/(alpha+gamma.prod()))  # 1
        exp1 = -np.pi**2 * torch.tensordot(t**2, Lq_inv.T, dims=1)  # Nx1
        exp2a = -torch.tensordot(np.pi**2/(alpha+gamma), (f-mean).T**2, dims=1)  # 1xM
        exp2b = -torch.tensordot(np.pi**2/(alpha+gamma), (f+mean).T**2, dims=1)  # 1xM
        exp3a = -2.0*np.pi * torch.tensordot(t.mm(Lq_inv), np.pi**2 * (f/alpha + mean/gamma).T, dims=1)  # NxM
        exp3b = -2.0*np.pi * torch.tensordot(t.mm(Lq_inv), np.pi**2 * (f/alpha - mean/gamma).T, dims=1)  # NxM

        a = 0.5 * sigma**2 * const * torch.exp(exp1)
        real = torch.exp(exp2a)*torch.cos(exp3a) + torch.exp(exp2b)*torch.cos(exp3b)
        imag = torch.exp(exp2a)*torch.sin(exp3a) + torch.exp(exp2b)*torch.sin(exp3b)
        return a * real, a * imag

    with torch.no_grad():
        Ktt = kernel(x)
        Ktt += model.likelihood.scale() * torch.eye(x.shape[0], device=gpr.config.device, dtype=gpr.config.dtype)
        Ltt = model._cholesky(Ktt, add_jitter=True)

        Kff = kernel_ff(w, w, kernel.sigma(), kernel.mean(), kernel.variance(), alpha)
        Pff = kernel_ff(w, -w, kernel.sigma(), kernel.mean(), kernel.variance(), alpha)
        Kff_real = 0.5 * (Kff + Pff)
        Kff_imag = 0.5 * (Kff - Pff)

        Ktf_real, Ktf_imag = kernel_tf(x, w, kernel.sigma(), kernel.mean(), kernel.variance(), alpha)

        a = torch.cholesky_solve(y,Ltt)
        b = torch.linalg.solve_triangular(Ltt,Ktf_real,upper=False)
        c = torch.linalg.solve_triangular(Ltt,Ktf_imag,upper=False)

        mu_real = Ktf_real.T.mm(a)
        mu_imag = Ktf_imag.T.mm(a)
        var_real = Kff_real - b.T.mm(b)
        var_imag = Kff_imag - c.T.mm(c)

        psd = mu_real**2 + mu_imag**2 + (var_real + var_imag).diagonal().reshape(-1,1)
        w = w.cpu().numpy()
        psd = psd.cpu().numpy()
    return w, psd
