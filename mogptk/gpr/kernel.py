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
        X1, X2 = self._check_input(X1, X2)
        return self.K(X1, X2)

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
        super().__setattr__(name, val)

    def _active_input(self, X1, X2=None):
        if self.active_dims is not None:
            X1 = torch.index_select(X1, dim=1, index=self.active_dims)
            if X2 is not None:
                X2 = torch.index_select(X2, dim=1, index=self.active_dims)
        return X1, X2

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
                raise ValueError("input dimensions for X1 and X2 must match")
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
                raise ValueError("must pass %d kernels" % length)
            for i in range(length - len(kernels)):
                kernels.append(copy.deepcopy(kernels[0]))
        for i, kernel in enumerate(kernels):
            if not issubclass(type(kernel), Kernel):
                raise ValueError("must pass kernels")
        if any(kernel.input_dims != kernels[0].input_dims for kernel in kernels[1:]):
            raise ValueError("kernels must have same input dimensions")
        if any(kernel.output_dims != kernels[0].output_dims for kernel in kernels[1:]):
            raise ValueError("kernels must have same output dimensions")
        if any(issubclass(type(kernel), MultiOutputKernel) != issubclass(type(kernels[0]), MultiOutputKernel) for kernel in kernels[1:]):
            raise ValueError("kernels must all inherit either Kernel or MultiOutputKernel")
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
                raise ValueError("input dimensions must match the number of active dimensions")
        self._active_dims = active_dims

    def iterkernels(self):
        yield self

    @property
    def output_dims(self):
        return 1

    def K(self, X1, X2=None):
        # X1 is NxD, X2 is MxD, then ret is NxM
        raise NotImplementedError()

    def K_diag(self, X1):
        # X1 is NxD, then ret is N
        # TODO: implement for all kernels
        return self.K(X1).diagonal()

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

    def __add__(self, other):
        return AddKernel(self, other)

    def __mul__(self, other):
        return MulKernel(self, other)

class Kernels(Kernel):
    def __init__(self, *kernels, name="Kernels"):
        super().__init__(name=name)
        kernels = self._check_kernels(kernels)

        i = 0
        while i < len(kernels):
            if isinstance(kernels[i], self.__class__):
                subkernels = kernels[i].kernels
                kernels = kernels[:i] + subkernels + kernels[i+1:]
                i += len(subkernels) - 1
            i += 1
        self.kernels = kernels

    def __getitem__(self, key):
        return self.kernels[key]

    def iterkernels(self):
        yield self
        for kernel in self.kernels:
            yield kernel

    @property
    def output_dims(self):
        # _check_kernels make sure it's the same for all kernels
        return self.kernels[0].output_dims

class AddKernel(Kernels):
    def __init__(self, *kernels, name="Add"):
        super().__init__(*kernels, name=name)

    def K(self, X1, X2=None):
        return torch.stack([kernel(X1, X2) for kernel in self.kernels], dim=2).sum(dim=2)

    def K_diag(self, X1):
        return torch.stack([kernel.K_diag(X1) for kernel in self.kernels], dim=1).sum(dim=1)

class MulKernel(Kernels):
    def __init__(self, *kernels, name="Mul"):
        super().__init__(*kernels, name=name)

    def K(self, X1, X2=None):
        return torch.stack([kernel(X1, X2) for kernel in self.kernels], dim=2).prod(dim=2)

    def K_diag(self, X1):
        return torch.stack([kernel.K_diag(X1) for kernel in self.kernels], dim=1).prod(dim=1)

class MixtureKernel(AddKernel):
    def __init__(self, kernels, Q, name="Mixture"):
        kernels = self._check_kernels(kernels, Q)
        super().__init__(*kernels, name=name)

class AutomaticRelevanceDeterminationKernel(MulKernel):
    def __init__(self, kernels, input_dims, name="ARD"):
        kernels = self._check_kernels(kernels, input_dims)
        for i, kernel in enumerate(kernels):
            kernel.set_active_dims(i)
        super().__init__(*kernels, name=name)

class MultiOutputKernel(Kernel):
    # The MultiOutputKernel is a base class for multi output kernels. It assumes that the first dimension of X contains channel IDs (integers) and calculate the final kernel matrix accordingly. Concretely, it will call the Ksub method for derived kernels from this class, which should return the kernel matrix between channel i and j, given inputs X1 and X2. This class will automatically split and recombine the input vectors and kernel matrices respectively, in order to create the final kernel matrix of the multi output kernel.
    # Be aware that for implementation of Ksub, i==j is true for the diagonal matrices. X2==None is true when calculating the Gram matrix (i.e. X1==X2) and when i==j. It is thus a subset of the case i==j, and if X2==None than i is always equal to j.
    # TODO: seems to accumulate a lot of memory in the loops to call Ksub, perhaps it's keeping the computational graph while indexing?

    def __init__(self, output_dims, input_dims=None, active_dims=None, name=None):
        super().__init__(input_dims, active_dims, name)
        self._output_dims = output_dims

    @property
    def output_dims(self):
        return self._output_dims

    def K(self, X1, X2=None):
        # X has shape (data_points,1+input_dims) where the first column is the channel ID

        # extract channel mask, get data, and find indices that belong to the channels
        c1 = X1[:,0].long()
        m1 = [c1==i for i in range(self.output_dims)]
        x1 = [X1[m1[i],1:] for i in range(self.output_dims)]
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
        else:
            # extract channel mask, get data, and find indices that belong to the channels
            c2 = X2[:,0].long()
            m2 = [c2==j for j in range(self.output_dims)]
            x2 = [X2[m2[j],1:] for j in range(self.output_dims)]
            r2 = [torch.nonzero(m2[j], as_tuple=False).reshape(1,-1) for j in range(self.output_dims)]  # as_tuple avoids warning

            res = torch.empty(X1.shape[0], X2.shape[0], device=config.device, dtype=config.dtype)  # N1 x N2
            for i in range(self.output_dims):
                for j in range(self.output_dims):
                    # calculate sub kernel matrix and add to main kernel matrix
                    res[r1[i],r2[j]] = self.Ksub(i, j, x1[i], x2[j])

        return res

    def K_diag(self, X1):
        # extract channel mask, get data, and find indices that belong to the channels
        c1 = X1[:,0].long()
        m1 = [c1==i for i in range(self.output_dims)]
        x1 = [X1[m1[i],1:] for i in range(self.output_dims)]  # I is broadcastable with last dimension in X
        r1 = [torch.nonzero(m1[i], as_tuple=False)[:,0] for i in range(self.output_dims)]  # as_tuple avoids warning

        res = torch.empty(X1.shape[0], device=config.device, dtype=config.dtype)  # N1 x N1

        # calculate lower triangle of main kernel matrix, the upper triangle is a transpose
        for i in range(self.output_dims):
            # calculate sub kernel matrix and add to main kernel matrix
            res[r1[i]] = self.Ksub_diag(i, x1[i])
        return res

    def Ksub(self, i, j, X1, X2=None):
        raise NotImplementedError()

    def Ksub_diag(self, i, X1):
        return self.Ksub(i, i, X1).diagonal()
