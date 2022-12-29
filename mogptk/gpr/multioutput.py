import torch
import numpy as np
from . import MultiOutputKernel, Parameter, config

class IndependentMultiOutputKernel(MultiOutputKernel):
    """
    Kernel with subkernels for each channels independently. Only the subkernels as block matrices on the diagonal are calculated, there is no correlation between channels.

    Args:
        kernels (list of Kernel): Kernels of shape (output_dims,).
        output_dims (int): Number of output dimensions.
        name (str): Kernel name.
    """
    def __init__(self, *kernels, output_dims=None, name="IMO"):
        if output_dims is None:
            output_dims = len(kernels)
        super().__init__(output_dims, name=name)
        self.kernels = self._check_kernels(kernels, output_dims)

    def __getitem__(self, key):
        return self.kernels[key]
    
    def Ksub(self, i, j, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1, X2 = self._active_input(X1, X2)
        if i == j:
            return self.kernels[i].K(X1, X2)
        else:
            if X2 is None:
                X2 = X1
            return torch.zeros(X1.shape[0], X2.shape[0], device=config.device, dtype=config.dtype)

    def Ksub_diag(self, i, X1):
        # X has shape (data_points,input_dims)
        X1, _ = self._active_input(X1)
        return self.kernels[i].K_diag(X1)

class MultiOutputSpectralKernel(MultiOutputKernel):
    """
    Multi-output spectral kernel (MOSM) where each channel and cross-channel is modelled with a spectral kernel as proposed by [1]. You can add the mixture kernel with `MixtureKernel(MultiOutputSpectralKernel(...), Q=3)`.

    $$ K_{ij}(x,x') = \\alpha_{ij} \\exp\\left(-\\frac{1}{2}(\\tau+\\theta_{ij})^T\\Sigma_{ij}(\\tau+\\theta_{ij})\\right) \\cos((\\tau+\\theta_{ij})^T\\mu_{ij} + \\phi_{ij}) $$

    $$ \\alpha_{ij} = w_{ij}\\sqrt{\\left((2\\pi)^n|\\Sigma_{ij}|\\right)} $$

    $$ w_{ij} = w_iw_j\\exp\\left(-\\frac{1}{4}(\\mu_i-\\mu_j)^T(\\Sigma_i+\\Sigma_j)^{-1}(\\mu_i-\\mu_j)\\right) $$

    $$ \\mu_{ij} = (\\Sigma_i+\\Sigma_j)^{-1}(\\Sigma_i\\mu_j + \\Sigma_j\\mu_i) $$

    $$ \\Sigma_{ij} = 2\\Sigma_i(\\Sigma_i+\\Sigma_j)^{-1}\\Sigma_j$$

    with \\(\\theta_{ij} = \\theta_i-\\theta_j\\), \\(\\phi_{ij} = \\phi_i - \\phi_j\\), \\(\\tau = |x-x'|\\), \\(w\\) the weight, \\(\\mu\\) the mean, \\(\\Sigma\\) the variance, \\(\\theta\\) the delay, and \\(\\phi\\) the phase.

    Args:
        output_dims (int): Number of output dimensions.
        input_dims (int): Number of input dimensions.
        active_dims (list of int): Indices of active dimensions of shape (input_dims,).
        name (str): Kernel name.

    Attributes:
        weight (mogptk.gpr.parameter.Parameter): Weight \\(w\\) of shape (output_dims,).
        mean (mogptk.gpr.parameter.Parameter): Mean \\(\\mu\\) of shape (output_dims,input_dims).
        variance (mogptk.gpr.parameter.Parameter): Variance \\(\\Sigma\\) of shape (output_dims,input_dims).
        delay (mogptk.gpr.parameter.Parameter): Delay \\(\\theta\\) of shape (output_dims,input_dims).
        phase (mogptk.gpr.parameter.Parameter): Phase \\(\\phi\\) in hertz of shape (output_dims,).

    [1] G. Parra and F. Tobar, "Spectral Mixture Kernels for Multi-Output Gaussian Processes", Advances in Neural Information Processing Systems 31, 2017
    """
    def __init__(self, output_dims, input_dims=1, active_dims=None, name="MOSM"):
        super().__init__(output_dims, input_dims, active_dims, name)

        # TODO: incorporate mixtures?
        # TODO: allow different input_dims per channel
        weight = torch.ones(output_dims)
        mean = torch.zeros(output_dims, input_dims)
        variance = torch.ones(output_dims, input_dims)
        delay = torch.zeros(output_dims, input_dims)
        phase = torch.zeros(output_dims)

        self.weight = Parameter(weight, lower=config.positive_minimum)
        self.mean = Parameter(mean, lower=config.positive_minimum)
        self.variance = Parameter(variance, lower=config.positive_minimum)
        self.delay = Parameter(delay)
        self.phase = Parameter(phase)
        if output_dims == 1:
            self.delay.train = False
            self.phase.train = False

        self.twopi = np.power(2.0*np.pi,float(input_dims)/2.0)

    def Ksub(self, i, j, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1, X2 = self._active_input(X1, X2)
        tau = self.distance(X1,X2)  # NxMxD
        if i == j:
            variance = self.variance()[i]
            alpha = self.weight()[i]**2 * self.twopi * variance.prod().sqrt()  # scalar
            exp = torch.exp(-0.5*torch.tensordot(tau**2, variance, dims=1))  # NxM
            cos = torch.cos(2.0*np.pi * torch.tensordot(tau, self.mean()[i], dims=1))  # NxM
            return alpha * exp * cos
        else:
            inv_variances = 1.0/(self.variance()[i] + self.variance()[j])  # D

            diff_mean = self.mean()[i] - self.mean()[j]  # D
            magnitude = self.weight()[i]*self.weight()[j]*torch.exp(-np.pi**2 * diff_mean.dot(inv_variances*diff_mean))  # scalar

            mean = inv_variances * (self.variance()[i]*self.mean()[j] + self.variance()[j]*self.mean()[i])  # D
            variance = 2.0 * self.variance()[i] * inv_variances * self.variance()[j]  # D
            delay = self.delay()[i] - self.delay()[j]  # D
            phase = self.phase()[i] - self.phase()[j]  # scalar

            alpha = magnitude * self.twopi * variance.prod().sqrt()  # scalar
            exp = torch.exp(-0.5 * torch.tensordot((tau+delay)**2, variance, dims=1))  # NxM
            cos = torch.cos(2.0*np.pi * (torch.tensordot(tau+delay, mean, dims=1) + phase))  # NxM
            return alpha * exp * cos

    def Ksub_diag(self, i, X1):
        # X has shape (data_points,input_dims)
        variance = self.variance()[i]
        alpha = self.weight()[i]**2 * self.twopi * variance.prod().sqrt()  # scalar
        return alpha.repeat(X1.shape[0])

class MultiOutputSpectralMixtureKernel(MultiOutputKernel):
    """
    Multi-output spectral mixture kernel (MOSM) where each channel and cross-channel is modelled with a spectral kernel as proposed by [1].

    $$ K_{ij}(x,x') = \\sum_{q=0}^Q\\alpha_{ijq} \\exp\\left(-\\frac{1}{2}(\\tau+\\theta_{ijq})^T\\Sigma_{ijq}(\\tau+\\theta_{ijq})\\right) \\cos((\\tau+\\theta_{ijq})^T\\mu_{ijq} + \\phi_{ijq}) $$

    $$ \\alpha_{ijq} = w_{ijq}\\sqrt{\\left((2\\pi)^n|\\Sigma_{ijq}|\\right)} $$

    $$ w_{ijq} = w_{iq}w_{jq}\\exp\\left(-\\frac{1}{4}(\\mu_{iq}-\\mu_{jq})^T(\\Sigma_{iq}+\\Sigma_{jq})^{-1}(\\mu_{iq}-\\mu_{jq})\\right) $$

    $$ \\mu_{ijq} = (\\Sigma_{iq}+\\Sigma_{jq})^{-1}(\\Sigma_{iq}\\mu_{jq} + \\Sigma_{jq}\\mu_{iq}) $$

    $$ \\Sigma_{ijq} = 2\\Sigma_{iq}(\\Sigma_{iq}+\\Sigma_{jq})^{-1}\\Sigma_{jq}$$

    with \\(\\theta_{ijq} = \\theta_{iq}-\\theta_{jq}\\), \\(\\phi_{ijq} = \\phi_{iq}-\\phi_{jq}\\), \\(\\tau = |x-x'|\\), \\(w\\) the weight, \\(\\mu\\) the mean, \\(\\Sigma\\) the variance, \\(\\theta\\) the delay, and \\(\\phi\\) the phase.

    Args:
        Q (int): Number mixture components.
        output_dims (int): Number of output dimensions.
        input_dims (int): Number of input dimensions.
        active_dims (list of int): Indices of active dimensions of shape (input_dims,).
        name (str): Kernel name.

    Attributes:
        weight (mogptk.gpr.parameter.Parameter): Weight \\(w\\) of shape (output_dims,Q).
        mean (mogptk.gpr.parameter.Parameter): Mean \\(\\mu\\) of shape (output_dims,Q,input_dims).
        variance (mogptk.gpr.parameter.Parameter): Variance \\(\\Sigma\\) of shape (output_dims,Q,input_dims).
        delay (mogptk.gpr.parameter.Parameter): Delay \\(\\theta\\) of shape (output_dims,Q,input_dims).
        phase (mogptk.gpr.parameter.Parameter): Phase \\(\\phi\\) in hertz of shape (output_dims,Q).

    [1] G. Parra and F. Tobar, "Spectral Mixture Kernels for Multi-Output Gaussian Processes", Advances in Neural Information Processing Systems 31, 2017
    """
    def __init__(self, Q, output_dims, input_dims=1, active_dims=None, name="MOSM"):
        super().__init__(output_dims, input_dims, active_dims, name)

        # TODO: allow different input_dims per channel
        weight = torch.ones(output_dims, Q)
        mean = torch.zeros(output_dims, Q, input_dims)
        variance = torch.ones(output_dims, Q, input_dims)
        delay = torch.zeros(output_dims, Q, input_dims)
        phase = torch.zeros(output_dims, Q)

        self.input_dims = input_dims
        self.weight = Parameter(weight, lower=config.positive_minimum)
        self.mean = Parameter(mean, lower=config.positive_minimum)
        self.variance = Parameter(variance, lower=config.positive_minimum)
        self.delay = Parameter(delay)
        self.phase = Parameter(phase)
        if output_dims == 1:
            self.delay.train = False
            self.phase.train = False

        self.twopi = np.power(2.0*np.pi,float(self.input_dims)/2.0)

    def Ksub(self, i, j, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1, X2 = self._active_input(X1, X2)
        tau = self.distance(X1,X2)  # NxMxD
        if i == j:
            variance = self.variance()[i]  # QxD
            alpha = self.weight()[i]**2 * self.twopi * variance.prod(dim=1).sqrt()  # Q
            exp = torch.exp(-0.5 * torch.einsum("nmd,qd->qnm", tau**2, variance))  # QxNxM
            cos = torch.cos(2.0*np.pi * torch.einsum("nmd,qd->qnm", tau, self.mean()[i]))  # QxNxM
            Kq = alpha[:,None,None] * exp * cos  # QxNxM
        else:
            inv_variances = 1.0/(self.variance()[i] + self.variance()[j])  # QxD

            diff_mean = self.mean()[i] - self.mean()[j]  # QxD
            magnitude = self.weight()[i]*self.weight()[j]*torch.exp(-np.pi**2 * torch.sum(diff_mean*inv_variances*diff_mean, dim=1))  # Q

            mean = inv_variances * (self.variance()[i]*self.mean()[j] + self.variance()[j]*self.mean()[i])  # QxD
            variance = 2.0 * self.variance()[i] * inv_variances * self.variance()[j]  # QxD
            delay = self.delay()[i] - self.delay()[j]  # QxD
            phase = self.phase()[i] - self.phase()[j]  # Q

            alpha = magnitude * self.twopi * variance.prod(dim=1).sqrt()  # Q
            tau_delay = tau[None,:,:,:] + delay[:,None,None,:]  # QxNxMxD
            exp = torch.exp(-0.5 * torch.einsum("qnmd,qd->qnm", (tau_delay)**2, variance))  # QxNxM
            cos = torch.cos(2.0*np.pi * (torch.einsum("qnmd,qd->qnm", tau_delay, mean) + phase[:,None,None]))  # QxNxM
            Kq = alpha[:,None,None] * exp * cos  # QxNxM
        return torch.sum(Kq, dim=0)

    def Ksub_diag(self, i, X1):
        # X has shape (data_points,input_dims)
        variance = self.variance()[i]
        alpha = self.weight()[i]**2 * self.twopi * variance.prod(dim=1).sqrt()  # Q
        return torch.sum(alpha).repeat(X1.shape[0])

class UncoupledMultiOutputSpectralKernel(MultiOutputKernel):
    """
    Uncoupled multi-output spectral kernel (uMOSM) where each channel and cross-channel is modelled with a spectral kernel. It is similar to the MOSM kernel but instead of training a weight per channel, we train the lower triangular of the weight between all channels. You can add the mixture kernel with `MixtureKernel(UncoupledMultiOutputSpectralKernel(...), Q=3)`.

    $$ K_{ij}(x,x') = \\alpha_{ij} \\exp\\left(-\\frac{1}{2}(\\tau+\\theta_{ij})^T\\Sigma_{ij}(\\tau+\\theta_{ij})\\right) \\cos((\\tau+\\theta_{ij})^T\\mu_{ij} + \\phi_{ij}) $$

    $$ \\alpha_{ij} = w_{ij}\\sqrt{\\left((2\\pi)^n|\\Sigma_{ij}|\\right)} $$

    $$ w_{ij} = \\sigma_{ij}^2\\exp\\left(-\\frac{1}{4}(\\mu_i-\\mu_j)^T(\\Sigma_i+\\Sigma_j)^{-1}(\\mu_i-\\mu_j)\\right) $$

    $$ \\mu_{ij} = (\\Sigma_i+\\Sigma_j)^{-1}(\\Sigma_i\\mu_j + \\Sigma_j\\mu_i) $$

    $$ \\Sigma_{ij} = 2\\Sigma_i(\\Sigma_i+\\Sigma_j)^{-1}\\Sigma_j$$

    with \\(\\theta_{ij} = \\theta_i-\\theta_j\\), \\(\\phi_{ij} = \\phi_i - \\phi_j\\), \\(\\tau = |x-x'|\\), \\(\\sigma\\) the weight, \\(\\mu\\) the mean, \\(\\Sigma\\) the variance, \\(\\theta\\) the delay, and \\(\\phi\\) the phase.

    Args:
        output_dims (int): Number of output dimensions.
        input_dims (int): Number of input dimensions.
        active_dims (list of int): Indices of active dimensions of shape (input_dims,).
        name (str): Kernel name.

    Attributes:
        weight (mogptk.gpr.parameter.Parameter): Weight \\(w\\) of the lower-triangular of shape (output_dims,output_dims).
        mean (mogptk.gpr.parameter.Parameter): Mean \\(\\mu\\) of shape (output_dims,input_dims).
        variance (mogptk.gpr.parameter.Parameter): Variance \\(\\Sigma\\) of shape (output_dims,input_dims).
        delay (mogptk.gpr.parameter.Parameter): Delay \\(\\theta\\) of shape (output_dims,input_dims).
        phase (mogptk.gpr.parameter.Parameter): Phase \\(\\phi\\) in hertz of shape (output_dims,).
    """
    def __init__(self, output_dims, input_dims=1, active_dims=None, name="uMOSM"):
        super().__init__(output_dims, input_dims, active_dims, name)

        weight = torch.ones(output_dims, output_dims).tril()
        mean = torch.zeros(output_dims, input_dims)
        variance = torch.ones(output_dims, input_dims)
        delay = torch.zeros(output_dims, input_dims)
        phase = torch.zeros(output_dims)

        self.weight = Parameter(weight)
        self.weight.num_parameters = int((output_dims*output_dims+output_dims)/2)
        self.mean = Parameter(mean, lower=config.positive_minimum)
        self.variance = Parameter(variance, lower=config.positive_minimum)
        self.delay = Parameter(delay)
        self.phase = Parameter(phase)
        if output_dims == 1:
            self.delay.train = False
            self.phase.train = False

        self.twopi = np.power(2.0*np.pi,float(input_dims)/2.0)

    def Ksub(self, i, j, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1, X2 = self._active_input(X1, X2)
        tau = self.distance(X1,X2)  # NxMxD
        magnitude = self.weight().tril().mm(self.weight().tril().T)
        if i == j:
            variance = self.variance()[i]
            alpha = magnitude[i,i] * self.twopi * variance.prod().sqrt()  # scalar
            exp = torch.exp(-0.5*torch.tensordot(tau**2, variance, dims=1))  # NxM
            cos = torch.cos(2.0*np.pi * torch.tensordot(tau, self.mean()[i], dims=1))  # NxM
            return alpha * exp * cos
        else:
            inv_variances = 1.0/(self.variance()[i] + self.variance()[j])  # D

            diff_mean = self.mean()[i] - self.mean()[j]  # D
            magnitude = magnitude[i,j] * torch.exp(-np.pi**2 * diff_mean.dot(inv_variances*diff_mean))  # scalar

            mean = inv_variances * (self.variance()[i]*self.mean()[j] + self.variance()[j]*self.mean()[i])  # D
            variance = 2.0 * self.variance()[i] * inv_variances * self.variance()[j]  # D
            delay = self.delay()[i] - self.delay()[j]  # D
            phase = self.phase()[i] - self.phase()[j]  # scalar

            alpha = magnitude * self.twopi * variance.prod().sqrt()  # scalar
            exp = torch.exp(-0.5 * torch.tensordot((tau+delay)**2, variance, dims=1))  # NxM
            cos = torch.cos(2.0*np.pi * torch.tensordot(tau+delay, mean, dims=1) + phase)  # NxM
            return alpha * exp * cos

    def Ksub_diag(self, i, X1):
        # X has shape (data_points,input_dims)
        magnitude = self.weight().tril().mm(self.weight().tril().T)
        variance = self.variance()[i]
        alpha = magnitude[i,i] * self.twopi * variance.prod().sqrt()  # scalar
        return alpha.repeat(X1.shape[0])

class MultiOutputHarmonizableSpectralKernel(MultiOutputKernel):
    """
    Multi-output harmonizable spectral kernel (MOHSM) where each channel and cross-channel is modelled with a spectral kernel as proposed by [1]. You can add the mixture kernel with `MixtureKernel(MultiOutputHarmonizableSpectralKernel(...), Q=3)`.

    $$ K_{ij}(x,x') = \\alpha_{ij} \\exp\\left(-\\frac{1}{2}(\\tau+\\theta_{ij})^T\\Sigma_{ij}(\\tau+\\theta_{ij})\\right) \\cos((\\tau+\\theta_{ij})^T\\mu_{ij} + \\phi) \\exp\\left(-\\frac{l_{ij}}{2}|\\bar{x}-c|\\right) $$

    $$ \\alpha_{ij} = w_{ij}(2\\pi)^n\\sqrt{\\left(|\\Sigma_{ij}|\\right)} $$

    $$ w_{ij} = w_iw_j\\exp\\left(-\\frac{1}{4}(\\mu_i-\\mu_j)^T(\\Sigma_i+\\Sigma_j)^{-1}(\\mu_i-\\mu_j)\\right) $$

    $$ \\mu_{ij} = (\\Sigma_i+\\Sigma_j)^{-1}(\\Sigma_i\\mu_j + \\Sigma_j\\mu_i) $$

    $$ \\Sigma_{ij} = 2\\Sigma_i(\\Sigma_i+\\Sigma_j)^{-1}\\Sigma_j$$

    with \\(\\theta_{ij} = \\theta_i-\\theta_j\\), \\(\\phi_{ij} = \\phi_i - \\phi_j\\), \\(\\tau = |x-x'|\\), \\(\\bar{x} = |x+x'|/2\\), \\(w\\) the weight, \\(\\mu\\) the mean, \\(\\Sigma\\) the variance, \\(l\\) the lengthscale, \\(c\\) the center, \\(\\theta\\) the delay, and \\(\\phi\\) the phase.

    Args:
        output_dims (int): Number of output dimensions.
        input_dims (int): Number of input dimensions.
        active_dims (list of int): Indices of active dimensions of shape (input_dims,).
        name (str): Kernel name.

    Attributes:
        weight (mogptk.gpr.parameter.Parameter): Weight \\(w\\) of shape (output_dims,).
        mean (mogptk.gpr.parameter.Parameter): Mean \\(\\mu\\) of shape (output_dims,input_dims).
        variance (mogptk.gpr.parameter.Parameter): Variance \\(\\Sigma\\) of shape (output_dims,input_dims).
        lengthscale (mogptk.gpr.parameter.Parameter): Lengthscale \\(l\\) of shape (output_dims,).
        center (mogptk.gpr.parameter.Parameter): Center \\(c\\) of shape (input_dims,).
        delay (mogptk.gpr.parameter.Parameter): Delay \\(\\theta\\) of shape (output_dims,input_dims).
        phase (mogptk.gpr.parameter.Parameter): Phase \\(\\phi\\) in hertz of shape (output_dims,).

    [1] M. Altamirano, "Nonstationary Multi-Output Gaussian Processes via Harmonizable Spectral Mixtures, 2021
    """
    def __init__(self, output_dims, input_dims=1, active_dims=None, name="MOHSM"):
        super().__init__(output_dims, input_dims, active_dims, name)

        # TODO: incorporate mixtures?
        # TODO: allow different input_dims per channel
        weight = torch.ones(output_dims)
        mean = torch.zeros(output_dims, input_dims)
        variance = torch.ones(output_dims, input_dims)
        lengthscale = torch.ones(output_dims)
        center = torch.zeros(input_dims)
        delay = torch.zeros(output_dims, input_dims)
        phase = torch.zeros(output_dims)

        self.weight = Parameter(weight, lower=config.positive_minimum)
        self.mean = Parameter(mean, lower=config.positive_minimum)
        self.variance = Parameter(variance, lower=config.positive_minimum)
        self.lengthscale = Parameter(lengthscale, lower=config.positive_minimum)
        self.center = Parameter(center)
        self.delay = Parameter(delay)
        self.phase = Parameter(phase)
        if output_dims == 1:
            self.delay.train = False
            self.phase.train = False

        self.twopi = np.power(2.0*np.pi, float(self.input_dims))

    def Ksub(self, i, j, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1, X2 = self._active_input(X1, X2)
        tau = self.distance(X1,X2)  # NxMxD
        avg = self.average(X1,X2)  # NxMxD

        if i == j:
            variance = self.variance()[i]
            lengthscale = self.lengthscale()[i]**2

            alpha = self.weight()[i]**2 * self.twopi * variance.prod().sqrt() * torch.pow(lengthscale.sqrt(), float(self.input_dims))  # scalar
            exp1 = torch.exp(-0.5 * torch.tensordot(tau**2, variance, dims=1))  # NxM
            exp2 = torch.exp(-0.5 * torch.tensordot((avg-self.center())**2, lengthscale*torch.ones(self.input_dims, device=config.device, dtype=config.dtype), dims=1))  # NxM
            cos = torch.cos(2.0 * np.pi * torch.tensordot(tau, self.mean()[i], dims=1))  # NxM
            return alpha * exp1 * cos * exp2
        else:
            lengthscale_i = self.lengthscale()[i]**2
            lengthscale_j = self.lengthscale()[j]**2
            inv_variances = 1.0/(self.variance()[i] + self.variance()[j])  # D
            inv_lengthscale = 1.0/(lengthscale_i + lengthscale_j)  # D
            diff_mean = self.mean()[i] - self.mean()[j]  # D

            magnitude = self.weight()[i]*self.weight()[j] * torch.exp(-np.pi**2 * diff_mean.dot(inv_variances*diff_mean))  # scalar
            mean = inv_variances * (self.variance()[i]*self.mean()[j] + self.variance()[j]*self.mean()[i])  # D
            variance = 2.0 * self.variance()[i] * inv_variances * self.variance()[j]  # D
            lengthscale = 2.0 * lengthscale_i * inv_lengthscale * lengthscale_j  # D
            delay = self.delay()[i] - self.delay()[j]  # D
            phase = self.phase()[i] - self.phase()[j]  # scalar

            alpha = magnitude * self.twopi * variance.prod().sqrt()*torch.pow(lengthscale.sqrt(),float(self.input_dims))  # scalar
            exp1 = torch.exp(-0.5 * torch.tensordot((tau+delay)**2, variance, dims=1))  # NxM
            exp2 = torch.exp(-0.5 * torch.tensordot((avg-self.center())**2, lengthscale*torch.ones(self.input_dims, device=config.device, dtype=config.dtype), dims=1))  # NxM
            cos = torch.cos(2.0 * np.pi * torch.tensordot(tau+delay, mean, dims=1) + phase)  # NxM
            return alpha * exp1 * cos * exp2

    def Ksub_diag(self, i, X1):
        # X has shape (data_points,input_dims)
        X1, _ = self._active_input(X1)
        variance = self.variance()[i]
        lengthscale = self.lengthscale()[i]**2
        alpha = self.weight()[i]**2 * self.twopi * variance.prod().sqrt() * torch.pow(lengthscale.sqrt(), float(self.input_dims))  # scalar
        exp2 = torch.exp(-0.5 * torch.tensordot((X1-self.center())**2, lengthscale*torch.ones(self.input_dims, device=config.device, dtype=config.dtype), dims=1))  # NxM
        return alpha * exp2

class CrossSpectralKernel(MultiOutputKernel):
    """
    Cross Spectral kernel as proposed by [1]. You can add the mixture kernel with `MixtureKernel(CrossSpectralKernel(...), Q=3)`.

    Args:
        output_dims (int): Number of output dimensions.
        input_dims (int): Number of input dimensions.
        Rq (int): Number of subcomponents.
        active_dims (list of int): Indices of active dimensions of shape (input_dims,).
        name (str): Kernel name.

    Attributes:
        amplitude (mogptk.gpr.parameter.Parameter): Amplitude \\(\\sigma^2\\) of shape (output_dims,Rq).
        mean (mogptk.gpr.parameter.Parameter): Mean \\(\\mu\\) of shape (input_dims,).
        variance (mogptk.gpr.parameter.Parameter): Variance \\(\\Sigma\\) of shape (input_dims,).
        shift (mogptk.gpr.parameter.Parameter): Shift \\(\\phi\\) of shape (output_dims,Rq).

    [1] K.R. Ulrich et al, "GP Kernels for Cross-Spectrum Analysis", Advances in Neural Information Processing Systems 28, 2015
    """
    def __init__(self, output_dims, input_dims=1, Rq=1, active_dims=None, name="CSM"):
        super().__init__(output_dims, input_dims, active_dims, name)

        amplitude = torch.ones(output_dims, Rq)
        mean = torch.zeros(input_dims)
        variance = torch.ones(input_dims)
        shift = torch.zeros(output_dims, Rq)

        self.amplitude = Parameter(amplitude, lower=config.positive_minimum)
        self.mean = Parameter(mean, lower=config.positive_minimum)
        self.variance = Parameter(variance, lower=config.positive_minimum)
        self.shift = Parameter(shift)

    def Ksub(self, i, j, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1, X2 = self._active_input(X1, X2)
        tau = self.distance(X1,X2)  # NxMxD
        if i == j:
            # put Rq into third dimension and sum at the end
            amplitude = self.amplitude()[i].reshape(1,1,-1)  # 1x1xRq
            exp = torch.exp(-0.5 * torch.tensordot(tau**2, self.variance(), dims=1)).unsqueeze(2)  # NxMx1
            # the following cos is as written in the paper, instead we take phi out of the product with the mean
            #cos = torch.cos(torch.tensordot(tau.unsqueeze(2), self.mean(), dims=1))
            cos = torch.cos(2.0*np.pi * torch.tensordot(tau, self.mean(), dims=1).unsqueeze(2)) # NxMxRq
            return torch.sum(amplitude * exp * cos, dim=2)
        else:
            shift = self.shift()[i] - self.shift()[j]  # Rq

            # put Rq into third dimension and sum at the end
            amplitude = torch.sqrt(self.amplitude()[i]*self.amplitude()[j]).reshape(1,1,-1)  # 1x1xRq
            exp = torch.exp(-0.5 * torch.tensordot(tau**2, self.variance(), dims=1)).unsqueeze(2)  # NxMx1
            # the following cos is as written in the paper, instead we take phi out of the product with the mean
            #cos = torch.cos(torch.tensordot(tau.unsqueeze(2) + shift.reshape(1,1,-1,1), self.mean(), dims=1))
            cos = torch.cos(2.0*np.pi * (torch.tensordot(tau, self.mean(), dims=1).unsqueeze(2) + shift.reshape(1,1,-1))) # NxMxRq
            return torch.sum(amplitude * exp * cos, dim=2)

    def Ksub_diag(self, i, X1):
        # X has shape (data_points,input_dims)
        amplitude = self.amplitude()[i].sum()
        return amplitude.repeat(X1.shape[0])

class LinearModelOfCoregionalizationKernel(MultiOutputKernel):
    """
    Linear model of coregionalization kernel (LMC) as proposed by [1].

    Args:
        kernels (list of Kernel): Kernels of shape (Q,).
        output_dims (int): Number of output dimensions.
        input_dims (int): Number of input dimensions.
        Q (int): Number of components.
        Rq (int): Number of subcomponents.
        name (str): Kernel name.

    Attributes:
        weight (mogptk.gpr.parameter.Parameter): Weight \\(w\\) of shape (output_dims,Q,Rq).

    [1] P. Goovaerts, "Geostatistics for Natural Resource Evaluation", Oxford University Press, 1997
    """
    def __init__(self, *kernels, output_dims, input_dims=1, Q=None, Rq=1, name="LMC"):
        super().__init__(output_dims, input_dims, name=name)

        if Q is None:
            Q = len(kernels)
        weight = torch.ones(output_dims, Q, Rq)

        self.kernels = self._check_kernels(kernels, Q)
        self.weight = Parameter(weight, lower=config.positive_minimum)

    def __getitem__(self, key):
        return self.kernels[key]

    def Ksub(self, i, j, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1, X2 = self._active_input(X1, X2)
        magnitude = torch.sum(self.weight()[i] * self.weight()[j], dim=1)  # Q
        kernels = torch.stack([kernel.K(X1,X2) for kernel in self.kernels], dim=2)  # NxMxQ
        return torch.tensordot(kernels, magnitude, dims=1)

    def Ksub_diag(self, i, X1):
        # X has shape (data_points,input_dims)
        X1, _ = self._active_input(X1)
        magnitude = torch.sum(self.weight()[i]**2, dim=1)  # Q
        kernels = torch.stack([kernel.K_diag(X1) for kernel in self.kernels], dim=1)  # NxQ
        return torch.tensordot(kernels, magnitude, dims=1)

class GaussianConvolutionProcessKernel(MultiOutputKernel):
    """
    Gaussian convolution process kernel (CONV) as proposed by [1].

    Args:
        output_dims (int): Number of output dimensions.
        input_dims (int): Number of input dimensions.
        active_dims (list of int): Indices of active dimensions of shape (input_dims,).
        name (str): Kernel name.

    Attributes:
        weight (mogptk.gpr.parameter.Parameter): Weight \\(w\\) of shape (output_dims,).
        variance (mogptk.gpr.parameter.Parameter): Variance \\(\\Sigma\\) of shape (output_dims,input_dims).
        base_variance (mogptk.gpr.parameter.Parameter): Base variance \\(\\Sigma_0\\) of shape (input_dims,).

    [1] M.A. Álvarez and N.D. Lawrence, "Sparse Convolved Multiple Output Gaussian Processes", Advances in Neural Information Processing Systems 21, 2009
    """
    def __init__(self, output_dims, input_dims=1, active_dims=None, name="CONV"):
        super().__init__(output_dims, input_dims, active_dims, name)

        weight = torch.ones(output_dims)
        variance = torch.ones(output_dims, input_dims)
        base_variance = torch.ones(input_dims)

        self.weight = Parameter(weight, lower=config.positive_minimum)
        self.variance = Parameter(variance, lower=0.0)
        self.base_variance = Parameter(base_variance, lower=config.positive_minimum)

    def Ksub(self, i, j, X1, X2=None):
        # X has shape (data_points,input_dims)
        X1, X2 = self._active_input(X1, X2)
        tau = self.squared_distance(X1,X2)  # NxMxD

        # differences with the thesis from Parra is that it lacks a multiplication of 2*pi, lacks a minus in the exponencial function, and doesn't write the variance matrices as inverted
        if X2 is None:
            variances = 2.0*self.variance()[i] + self.base_variance()  # D
            magnitude = self.weight()[i]**2 * torch.sqrt(self.base_variance().prod()/variances.prod())  # scalar
            exp = torch.exp(-0.5 * torch.tensordot(tau, 1.0/variances, dims=1))  # NxM
            return magnitude * exp
        else:
            variances = self.variance()[i] + self.variance()[j] + self.base_variance()  # D
            weight_variance = torch.sqrt(self.base_variance().prod()/variances.prod())  # scalar
            magnitude = self.weight()[i] * self.weight()[j] * weight_variance  # scalar
            exp = torch.exp(-0.5 * torch.tensordot(tau, 1.0/variances, dims=1))  # NxM
            return magnitude * exp

    def Ksub_diag(self, i, X1):
        # X has shape (data_points,input_dims)
        variances = 2.0*self.variance()[i] + self.base_variance()  # D
        magnitude = self.weight()[i]**2 * torch.sqrt(self.base_variance().prod()/variances.prod())  # scalar
        return magnitude.repeat(X1.shape[0])
