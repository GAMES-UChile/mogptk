import torch
import copy
from . import Parameter, config

class Kernel:
    def __init__(self, input_dims=None, active_dims=None, name=None):
        if name is None:
            name = self.__class__.__name__
            if name.endswith('Kernel') and name != 'Kernel':
                name = name[:-6]

        self.input_dims = input_dims
        self.active_dims = active_dims
        self.name = name

    def __call__(self, X1, X2=None):
        return self.K(X1,X2)

    def __setattr__(self, name, val):
        if name == 'trainable':
            from .util import _find_parameters
            for p in _find_parameters(self):
                p.trainable = val
            return
        if hasattr(self, name) and isinstance(getattr(self, name), Parameter):
            raise AttributeError("parameter is read-only, use Parameter.assign()")
        if isinstance(val, Parameter) and val.name is None:
            val.name = name
        super(Kernel,self).__setattr__(name, val)

    def _check_input(self, X1, X2=None):
        if len(X1.shape) != 2:
            raise ValueError("X should have two dimensions (data_points,input_dims)")
        if X1.shape[0] == 0 or X1.shape[1] == 0:
            raise ValueError("X must not be empty")
        if X2 is not None:
            if len(X2.shape) != 2:
                raise ValueError("X should have two dimensions (data_points,input_dims)")
            if X2.shape[0] == 0:
                raise ValueError("X must not be empty")
            if X1.shape[1] != X2.shape[1]:
                raise ValueError("input_dims for X1 and X2 must match")

        if self.active_dims is not None:
            X1 = torch.index_select(X1, dim=1, index=self.active_dims)
            if X2 is not None:
                X2 = torch.index_select(X2, dim=1, index=self.active_dims)

        return X1, X2

    def _check_kernels(self, kernels, length=None):
        if isinstance(kernels, tuple):
            if len(kernels) == 1 and isinstance(kernels[0], list):
                kernels = kernels[0]
            else:
                kernels = list(kernels)
        elif not isinstance(kernels, list):
            kernels = [kernels]
        if len(kernels) == 0:
            raise ValueError("must pass at least one kernel")
        elif length is not None and len(kernels) != length:
            if len(kernels) != 1:
                raise ValueError("must pass %d kernel" % length)
            for i in range(length - len(kernels)):
                kernels.append(copy.deepcopy(kernels[0]))
        for i, kernel in enumerate(kernels):
            if not issubclass(type(kernel), Kernel):
                raise ValueError("must pass kernels")
        return kernels

    @property
    def active_dims(self):
        return self._active_dims

    @active_dims.setter
    def active_dims(self, active_dims):
        if active_dims is not None:
            if not isinstance(active_dims, list):
                active_dims = [active_dims]
            if not all(isinstance(item, int) for item in active_dims):
                raise ValueError("active dimensions must be a list of integers")
            active_dims = torch.tensor(active_dims, device=config.device, dtype=torch.long)
            if self.input_dims is not None and self.input_dims != active_dims.shape[0]:
                raise ValueError("input dimensions must match the number of actived dimensions")
        self._active_dims = active_dims

    def K(self, X1, X2=None):
        raise NotImplementedError()

    def distance(self, X1, X2=None):
        # X1 is NxD, X2 is MxD, then ret is NxMxD
        if X2 is None:
            X2 = X1
        return X1.unsqueeze(1) - X2

    def squared_distance(self, X1, X2=None):
        # X1 is NxD, X2 is MxD, then ret is NxMxD
        if X2 is None:
            X2 = X1
        #return (X1.unsqueeze(1) - X2)**2  # slower than cdist for large X
        return torch.cdist(X2.T.unsqueeze(2), X1.T.unsqueeze(2)).T**2

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

class MultiOutputKernel(Kernel):
    # The MultiOutputKernel is a base class for multi output kernels. It assumes that the first dimension of X contains channel IDs (integers) and calculate the final kernel matrix accordingly. Concretely, it will call the Ksub method for derived kernels from this class, which should return the kernel matrix between channel i and j, given inputs X1 and X2. This class will automatically split and recombine the input vectors and kernel matrices respectively, in order to create the final kernel matrix of the multi output kernel.
    # Be aware that for implementation of Ksub, i==j is true for the diagonal matrices. X2==None is true when calculating the Gram matrix (i.e. X1==X2) and when i==j. It is thus a subset of the case i==j, and if X2==None than i is always equal to j.

    def __init__(self, output_dims, input_dims=None, active_dims=None, name=None):
        super(MultiOutputKernel, self).__init__(input_dims, active_dims, name)

        noise = torch.ones(output_dims)

        self.output_dims = output_dims
        self.noise = Parameter(noise, lower=config.positive_minimum)

    def K(self, X1, X2=None):
        # X has shape (data_points,1+input_dims) where the first column is the channel ID
        X1,X2 = self._check_input(X1,X2)

        # extract channel mask, get data, and find indices that belong to the channels
        I1 = X1[:,0].long()
        m1 = [I1==i for i in range(self.output_dims)]
        x1 = [X1[m1[i],1:] for i in range(self.output_dims)]  # I is broadcastable with last dimension in X
        r1 = [torch.nonzero(m1[i], as_tuple=False) for i in range(self.output_dims)]  # as_tuple avoids warning

        if X2 is None:
            r2 = [r1[i].reshape(1,-1) for i in range(self.output_dims)]
            res = torch.empty(X1.shape[0], X1.shape[0], device=config.device, dtype=config.dtype)  # N1 x N1
            # calculate lower triangle of main kernel matrix, the upper triangle is a transpose
            for i in range(self.output_dims):
                for j in range(i+1):
                    # calculate sub kernel matrix and add to main kernel matrix
                    if i == j:
                        res[r1[i],r2[i]] = self.Ksub(i, i, x1[i])
                    else:
                        k = self.Ksub(i, j, x1[i], x1[j])
                        res[r1[i],r2[j]] = k
                        res[r1[j],r2[i]] = k.T

            # add noise per channel
            res += torch.index_select(self.noise(), dim=0, index=I1).diagflat()
        else:
            # extract channel mask, get data, and find indices that belong to the channels
            I2 = X2[:,0].long()
            m2 = [I2==j for j in range(self.output_dims)]
            x2 = [X2[m2[j],1:] for j in range(self.output_dims)]  # I is broadcastable with last dimension in X
            r2 = [torch.nonzero(m2[j], as_tuple=False).reshape(1,-1) for j in range(self.output_dims)]  # as_tuple avoids warning

            res = torch.empty(X1.shape[0], X2.shape[0], device=config.device, dtype=config.dtype)  # N1 x N2
            for i in range(self.output_dims):
                for j in range(self.output_dims):
                    # calculate sub kernel matrix and add to main kernel matrix
                    res[r1[i],r2[j]] = self.Ksub(i, j, x1[i], x2[j])

        return res

    def Ksub(self, i, j, X1, X2=None):
        raise NotImplementedError()
