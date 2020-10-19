import torch
from . import Kernel

class AddKernel(Kernel):
    def __init__(self, *kernels, name="Add"):
        super(AddKernel, self).__init__(name=name)
        self.kernels = self._check_kernels(kernels)

    def __getitem__(self, key):
        return self.kernels[key]

    def K(self, X1, X2=None):
        return torch.stack([kernel(X1,X2) for kernel in self.kernels], dim=2).sum(dim=2)

class MulKernel(Kernel):
    def __init__(self, *kernels, name="Mul"):
        super(MulKernel, self).__init__(name=name)
        self.kernels = self._check_kernels(kernels)

    def __getitem__(self, key):
        return self.kernels[key]

    def K(self, X1, X2=None):
        return torch.stack([kernel(X1,X2) for kernel in self.kernels], dim=2).prod(dim=2)

class MixtureKernel(AddKernel):
    def __init__(self, kernel, Q, name="Mixture"):
        Kernel.__init__(self, name=name)
        self.kernels = self._check_kernels(kernel, Q)

class AutomaticRelevanceDeterminationKernel(MulKernel):
    def __init__(self, kernel, input_dims, name="ARD"):
        Kernel.__init__(self, name=name)
        self.kernels = self._check_kernels(kernel, input_dims)
        for i, kernel in enumerate(self.kernels):
            kernel.set_active_dims(i)
