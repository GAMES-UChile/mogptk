import torch
import copy
from . import Parameter, config

class Kernel:
    """
    Base kernel.

    Args:
        input_dims (int): Number of input dimensions.
        active_dims (list of int): Indices of active dimensions of shape (input_dims,).
        name (str): Kernel name.
    """
    def __init__(self, input_dims=None, active_dims=None, name=None):
        if name is None:
            name = self.__class__.__name__
            if name.endswith('Kernel') and name != 'Kernel':
                name = name[:-6]

        self.input_dims = input_dims
        self.active_dims = active_dims  # checks input
        self.output_dims = None
        self.name = name

    def __call__(self, X1, X2=None):
        """
        Calculate kernel matrix. This is the same as calling `K(X1,X2)` but `X1` and `X2` don't necessarily have to be tensors. If `X2` is not given, it is assumed to be the same as `X1`. Not passing `X2` may be faster for some kernels.

        Args:
            X1 (torch.tensor): Input of shape (data_points0,input_dims).
            X2 (torch.tensor): Input of shape (data_points1,input_dims).

        Returns:
            torch.tensor: Kernel matrix of shape (data_points0,data_points1).
        """
        X1, X2 = self._check_input(X1, X2)
        return self.K(X1, X2)

    def __setattr__(self, name, val):
        if name == 'train':
            from .util import _find_parameters
            for _, p in _find_parameters(self):
                p.train = val
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
        if not torch.is_tensor(X1):
            X1 = torch.tensor(X1, device=config.device, dtype=config.dtype)
        elif X1.device != config.device or X1.dtype != config.dtype:
            X1 = X1.to(device, dtype)
        if X1.ndim != 2:
            raise ValueError("X should have two dimensions (data_points,input_dims)")
        if X1.shape[0] == 0 or X1.shape[1] == 0:
            raise ValueError("X must not be empty")
        if X2 is not None:
            if not torch.is_tensor(X2):
                X2 = torch.tensor(X2, device=config.device, dtype=config.dtype)
            elif X2.device != config.device or X2.dtype != config.dtype:
                X2 = X2.to(device, dtype)
            if X2.ndim != 2:
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
        output_dims = [kernel.output_dims for kernel in kernels if kernel.output_dims is not None]
        if any(output_dim != output_dims[0] for output_dim in output_dims[1:]):
            raise ValueError("multi-output kernels must have same output dimensions")
        if len(output_dims) != 0:
            for kernel in kernels:
                if kernel.active_dims is None and kernel.output_dims is None:
                    input_dims = kernel.input_dims if kernel.input_dims is not None else 1
                    kernel.active_dims = [input_dim+1 for input_dim in range(input_dims)]
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
            self.input_dims = active_dims.shape[0]
        self._active_dims = active_dims

    def iterkernels(self):
        """
        Iterator of all kernels and subkernels. This is useful as some kernels are composed of other kernels (for example the `AddKernel`).
        """
        yield self

    def copy_parameters(self, other):
        """
        Copy parameters from other kernel.

        Args:
            other (Kernel): Source kernel.
        """
        from .util import _copy_parameters
        _copy_parameters(self, other)

    def K(self, X1, X2=None):
        """
        Calculate kernel matrix. If `X2` is not given, it is assumed to be the same as `X1`. Not passing `X2` may be faster for some kernels.

        Args:
            X1 (torch.tensor): Input of shape (data_points0,input_dims).
            X2 (torch.tensor): Input of shape (data_points1,input_dims).

        Returns:
            torch.tensor: Kernel matrix of shape (data_points0,data_points1).
        """
        # X1 is NxD, X2 is MxD, then ret is NxM
        raise NotImplementedError()

    def K_diag(self, X1):
        """
        Calculate the diagonal of the kernel matrix. This is usually faster than `K(X1).diagonal()`.

        Args:
            X1 (torch.tensor): Input of shape (data_points,input_dims).

        Returns:
            torch.tensor: Kernel matrix diagonal of shape (data_points,).
        """
        # X1 is NxD, then ret is N
        return self.K(X1).diagonal()

    @staticmethod
    def average(X1, X2=None):
        # X1 is NxD, X2 is MxD, then ret is NxMxD
        if X2 is None:
            X2 = X1
        return 0.5 * (X1.unsqueeze(1) + X2)

    @staticmethod
    def distance(X1, X2=None):
        # X1 is NxD, X2 is MxD, then ret is NxMxD
        if X2 is None:
            X2 = X1
        return X1.unsqueeze(1) - X2

    @staticmethod
    def squared_distance(X1, X2=None):
        # X1 is NxD, X2 is MxD, then ret is NxMxD
        if X2 is None:
            X2 = X1
        #return (X1.unsqueeze(1) - X2)**2  # slower than cdist for large X
        return torch.cdist(X2.T.unsqueeze(2), X1.T.unsqueeze(2)).permute((2,1,0))**2

    def __add__(self, other):
        return AddKernel(self, other)

    def __mul__(self, other):
        return MulKernel(self, other)

class Kernels(Kernel):
    """
    Base kernel for list of kernels.

    Args:
        kernels (list of Kernel): Kernels.
        name (str): Kernel name.
    """
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

        self.input_dims = kernels[0].input_dims
        output_dims = [kernel.output_dims for kernel in kernels if kernel.output_dims is not None]
        if len(output_dims) == 0:
            self.output_dims = None
        else:
            self.output_dims = output_dims[0]  # they are all equal

    def __getitem__(self, key):
        return self.kernels[key]

    def iterkernels(self):
        yield self
        for kernel in self.kernels:
            yield kernel

class AddKernel(Kernels):
    """
    Addition kernel that sums kernels.

    Args:
        kernels (list of Kernel): Kernels.
        name (str): Kernel name.
    """
    def __init__(self, *kernels, name="Add"):
        super().__init__(*kernels, name=name)

    def K(self, X1, X2=None):
        return torch.stack([kernel(X1, X2) for kernel in self.kernels], dim=2).sum(dim=2)

    def K_diag(self, X1):
        return torch.stack([kernel.K_diag(X1) for kernel in self.kernels], dim=1).sum(dim=1)

class MulKernel(Kernels):
    """
    Multiplication kernel that multiplies kernels.

    Args:
        kernels (list of Kernel): Kernels.
        name (str): Kernel name.
    """
    def __init__(self, *kernels, name="Mul"):
        super().__init__(*kernels, name=name)

    def K(self, X1, X2=None):
        return torch.stack([kernel(X1, X2) for kernel in self.kernels], dim=2).prod(dim=2)

    def K_diag(self, X1):
        return torch.stack([kernel.K_diag(X1) for kernel in self.kernels], dim=1).prod(dim=1)

class MixtureKernel(AddKernel):
    """
    Mixture kernel that sums `Q` kernels.

    Args:
        kernel (Kernel): Single kernel.
        Q (int): Number of mixtures.
        name (str): Kernel name.
    """
    def __init__(self, kernel, Q, name="Mixture"):
        if not issubclass(type(kernel), Kernel):
            raise ValueError("must pass kernel")
        kernels = self._check_kernels(kernel, Q)
        super().__init__(*kernels, name=name)

class AutomaticRelevanceDeterminationKernel(MulKernel):
    """
    Automatic relevance determination (ARD) kernel that multiplies kernels for each input dimension.

    Args:
        kernel (Kernel): Single kernel.
        input_dims (int): Number of input dimensions.
        name (str): Kernel name.
    """
    def __init__(self, kernel, input_dims, name="ARD"):
        if not issubclass(type(kernel), Kernel):
            raise ValueError("must pass kernel")
        kernels = self._check_kernels(kernel, input_dims)
        for i, kernel in enumerate(kernels):
            kernel.set_active_dims(i)
        super().__init__(*kernels, name=name)

class MultiOutputKernel(Kernel):
    """
    The `MultiOutputKernel` is a base class for multi-output kernels. It assumes that the first dimension of `X` contains channel IDs (integers) and calculates the final kernel matrix accordingly. Concretely, it will call the `Ksub` method for derived kernels from this class, which should return the kernel matrix between channel `i` and `j`, given inputs `X1` and `X2`. This class will automatically split and recombine the input vectors and kernel matrices respectively, in order to create the final kernel matrix of the multi-output kernel.

    Be aware that for implementation of `Ksub`, `i==j` is true for the diagonal matrices. `X2==None` is true when calculating the Gram matrix (i.e. `X1==X2`) and when `i==j`. It is thus a subset of the case `i==j`, and if `X2==None` than `i` is always equal to `j`.

    Args:
        output_dims (int): Number of output dimensions.
        input_dims (int): Number of input dimensions.
        active_dims (list of int): Indices of active dimensions of shape (input_dims,).
        name (str): Kernel name.
    """
    # TODO: seems to accumulate a lot of memory in the loops to call Ksub, perhaps it's keeping the computational graph while indexing?

    def __init__(self, output_dims, input_dims=None, active_dims=None, name=None):
        super().__init__(input_dims, active_dims, name)
        self.output_dims = output_dims

    def _check_input(self, X1, X2=None):
        X1, X2 = super()._check_input(X1, X2)
        if not torch.all(X1[:,0] == X1[:,0].long()) or not torch.all(X1[:,0] < self.output_dims):
            raise ValueError("X must have integers for the channel IDs in the first input dimension")
        if X2 is not None and not torch.all(X2[:,0] == X2[:,0].long()) or not torch.all(X1[:,0] < self.output_dims):
            raise ValueError("X must have integers for the channel IDs in the first input dimension")
        return X1, X2

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
        """
        Calculate kernel matrix between two channels. If `X2` is not given, it is assumed to be the same as `X1`. Not passing `X2` may be faster for some kernels.

        Args:
            X1 (torch.tensor): Input of shape (data_points0,input_dims).
            X2 (torch.tensor): Input of shape (data_points1,input_dims).

        Returns:
            torch.tensor: Kernel matrix of shape (data_points0,data_points1).
        """
        raise NotImplementedError()

    def Ksub_diag(self, i, X1):
        """
        Calculate the diagonal of the kernel matrix between two channels. This is usually faster than `Ksub(X1).diagonal()`.

        Args:
            X1 (torch.tensor): Input of shape (data_points,input_dims).

        Returns:
            torch.tensor: Kernel matrix diagonal of shape (data_points,).
        """
        return self.Ksub(i, i, X1).diagonal()
