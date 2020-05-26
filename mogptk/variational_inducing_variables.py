import numpy as np
import tensorflow as tf
import gpflow
from gpflow.inducing_variables import InducingVariables


class VariationalInducingFunctions(InducingVariables):
    def __init__(self, *args, input_dim=1, magnitude=None, variance=None):
        """
        Inducing variables for variational sparse MOSM
        using variational inducing kernels.

        z: inducing inputs, sequence of array_like
            The arrays must have the same shape (n, input_dim), except in the first dimension (n variable).

        with z[:, 0] the index of the Q
        and z[:, 1] the value of the inducing inputs

        Note: currently only tested for input_dim=1
        """

        if input_dim > 1:
            raise Exception('Not implemented for input_dim>1')

        self.Q = len(args)
        if magnitude is not None:
            assert magnitude.shape == (self.Q,)
        else:
            magnitude = np.random.random(self.Q)
                    
        if variance is not None:
            assert variance.shape == (input_dim, self.Q)
        else:
            variance = np.random.random((input_dim, self.Q))
        
        # latent component index
        z = np.concatenate(args, 0).reshape(-1, input_dim)
        
        self.z_index = np.repeat(np.arange(self.Q), [len(a) for a in args])
        self.K = int(z.shape[0]/self.Q)
        self.z = gpflow.Parameter(z, dtype=gpflow.default_float())
        self.magnitude = gpflow.Parameter(magnitude, dtype=gpflow.default_float())
        self.variance = gpflow.Parameter(
            variance,
            transform=gpflow.utilities.positive(),
            dtype=gpflow.default_float())
    
    def __len__(self):
        return self.z.shape[0]