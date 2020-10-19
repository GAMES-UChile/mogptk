import torch
import numpy as np
from . import MultiOutputKernel, Parameter, device, dtype, positive_minimum

class IndependentMultiOutputKernel(MultiOutputKernel):
    def __init__(self, *kernels, output_dims, name="IMO"):
        super(IndependentMultiOutputKernel, self).__init__(output_dims, name=name)

        kernels = self._check_kernels(kernels, output_dims)

        self.kernels = kernels

    def __getitem__(self, key):
        return self.kernels[key]
    
    def Ksub(self, i, j, X1, X2=None):
        # X has shape (data_points,input_dims)
        if i == j:
            return self.kernels[i](X1, X2)
        else:
            if X2 is None:
                X2 = X1
            return torch.zeros(X1.shape[0], X2.shape[0], device=device, dtype=dtype)

class MultiOutputSpectralKernel(MultiOutputKernel):
    def __init__(self, output_dims, input_dims, active_dims=None, name="MOSM"):
        super(MultiOutputSpectralKernel, self).__init__(output_dims, input_dims, active_dims, name)

        magnitude = torch.randn(output_dims)
        mean = torch.rand(output_dims, input_dims)
        variance = torch.rand(output_dims, input_dims)
        delay = torch.zeros(output_dims, input_dims)
        phase = torch.zeros(output_dims)

        self.input_dims = input_dims
        self.magnitude = Parameter(magnitude)
        self.mean = Parameter(mean, lower=positive_minimum)
        self.variance = Parameter(variance, lower=positive_minimum)
        if 1 < output_dims:
            self.delay = Parameter(delay)  # TODO: first delay must remain zero
            self.phase = Parameter(phase)  # TODO: first phase must remain zero

        self.twopi = np.power(2.0*np.pi,float(self.input_dims)/2.0)

    def Ksub(self, i, j, X1, X2=None):
        # X has shape (data_points,input_dims)
        tau = self.distance(X1,X2)  # NxMxD
        if i == j:
            variance = self.variance()[i]
            alpha = self.magnitude()[i]**2 * self.twopi * variance.prod().sqrt()  # scalar
            exp = torch.exp(-0.5*torch.tensordot(tau**2, variance, dims=1))  # NxM
            cos = torch.cos(torch.tensordot(tau, self.mean()[i], dims=1))  # NxM
            return alpha * exp * cos
        else:
            inv_variances = 1.0/(self.variance()[i] + self.variance()[j])  # D

            diff_mean = self.mean()[i] - self.mean()[j]  # D
            magnitude = self.magnitude()[i]*self.magnitude()[j]*torch.exp(-0.25 * diff_mean.dot(inv_variances*diff_mean))  # scalar

            mean = inv_variances * (self.variance()[i]*self.mean()[j] + self.variance()[j]*self.mean()[i])  # D
            variance = 2.0 * self.variance()[i] * inv_variances * self.variance()[j]  # D
            delay = self.delay()[i] - self.delay()[j]  # D
            phase = self.phase()[i] - self.phase()[j]  # scalar

            # TODO: more stable to do (squared) distance(X1+sqrt(delay),X2+sqrt(delay))?
            #print("MOSM", X1.shape, X2.shape, tau.shape)
            #print(self.squared_distance(X1,X2).shape)
            #print(torch.tensordot((tau+delay)**2, variance, dims=1))
            #print(torch.tensordot(variance, (tau+delay)**2, dims=1))
            alpha = magnitude * self.twopi * variance.prod().sqrt()  # scalar
            exp = torch.exp(-0.5 * torch.tensordot((tau+delay)**2, variance, dims=1))  # NxM
            cos = torch.cos(torch.tensordot(tau+delay, mean, dims=1) + phase)  # NxM
            return alpha * exp * cos

class CrossSpectralKernel(MultiOutputKernel):
    def __init__(self, output_dims, input_dims, Rq=1, active_dims=None, name="CSM"):
        super(CrossSpectralKernel, self).__init__(output_dims, input_dims, active_dims, name)

        amplitude = torch.rand(output_dims, Rq)
        mean = torch.rand(input_dims)
        variance = torch.rand(input_dims)
        shift = torch.zeros(output_dims, Rq)

        self.input_dims = input_dims
        self.Rq = Rq
        self.amplitude = Parameter(amplitude, lower=positive_minimum)
        self.mean = Parameter(mean, lower=positive_minimum)
        self.variance = Parameter(variance, lower=positive_minimum)
        self.shift = Parameter(shift)

    def Ksub(self, i, j, X1, X2=None):
        # X has shape (data_points,input_dims)
        if X2 is None:
            X2 = X1  # TODO

        tau = self.distance(X1,X2)  # NxMxD
        shift = self.shift()[i] - self.shift()[j]  # Rq

        # put Rq into third dimension and sum at the end
        amplitude = torch.sqrt(self.amplitude()[i]*self.amplitude()[j]).reshape(1,1,-1)  # 1x1xRq
        exp = torch.exp(-0.5 * torch.tensordot(tau**2, self.variance(), dims=1)).unsqueeze(2)  # NxMx1
        # the following cos is as written in the paper, the used one takes phi out of the product with the mean
        #cos = torch.cos(torch.tensordot(tau.unsqueeze(2) + shift.reshape(1,1,-1,1), self.mean(), dims=1))
        cos = torch.cos(torch.tensordot(tau, self.mean(), dims=1).unsqueeze(2) + shift.reshape(1,1,-1)) # NxMxRq
        return torch.sum(amplitude * exp * cos, dim=2)

class LinearModelOfCoregionalizationKernel(MultiOutputKernel):
    def __init__(self, *kernels, output_dims, input_dims, Rq=1, name="LMC"):
        super(LinearModelOfCoregionalizationKernel, self).__init__(output_dims, input_dims, name=name)

        kernels = self._check_kernels(kernels)
        weight = torch.rand(output_dims, len(kernels), Rq)

        self.kernels = kernels
        self.weight = Parameter(weight, lower=positive_minimum)

    def __getitem__(self, key):
        return self.kernels[key]

    def Ksub(self, i, j, X1, X2=None):
        # X has shape (data_points,input_dims)
        weight = torch.sum(self.weight()[i] * self.weight()[j], dim=1)  # K
        kernels = torch.stack([kernel(X1,X2) for kernel in self.kernels], dim=2)  # NxMxK
        return torch.tensordot(kernels, weight, dims=1)

class GaussianConvolutionProcessKernel(MultiOutputKernel):
    def __init__(self, output_dims, input_dims, active_dims=None, name="CONV"):
        super(GaussianConvolutionProcessKernel, self).__init__(output_dims, input_dims, active_dims, name)

        weight = torch.rand(output_dims)
        variance = torch.rand(output_dims, input_dims)
        base_variance = torch.rand(input_dims)

        self.input_dims = input_dims
        self.weight = Parameter(weight, lower=positive_minimum)
        self.variance = Parameter(variance, lower=positive_minimum)
        self.base_variance = Parameter(base_variance, lower=positive_minimum)

    def Ksub(self, i, j, X1, X2=None):
        # X has shape (data_points,input_dims)
        tau = self.distance(X1,X2)  # NxMxD
        inv_base_variance = 1.0/self.base_variance()  # D
        if X2 is None:
            variances = 2.0/self.variance()[i] + inv_base_variance  # D
            weight = self.weight()[i]**2 * torch.sqrt(inv_base_variance.prod()/variances.prod())  # scalar
            exp = torch.exp(-0.5 * torch.tensordot(tau**2, 1.0/variances, dims=1))  # NxM
            return weight * exp
        else:
            variances = 1.0/self.variance()[i] + 1.0/self.variance()[j] + inv_base_variance  # D
            weight_variance = torch.sqrt(inv_base_variance.prod()/variances.prod())  # scalar
            weight = self.weight()[i] * self.weight()[j] * weight_variance  # scalar
            exp = torch.exp(-0.5 * torch.tensordot(tau**2, 1.0/variances, dims=1))  # NxM
            return weight * exp
