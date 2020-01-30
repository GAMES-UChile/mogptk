import gpflow
import numpy as np
import tensorflow as tf

from gpflow.kernels import Kernel

class Noise(Kernel):
    def __init__(self, input_dim, output_dim, active_dims=None):
        Kernel.__init__(self, active_dims)
        self.input_dim = input_dim
        self.output_dim = output_dim

        noise = np.random.random((output_dim))
        self.noise = gpflow.Parameter(noise, transform=gpflow.utilities.positive(), name="noise")

    def K(self, X, X2=None, presliced=False):
        if X2 is not None and not np.array_equal(X, X2):
            return tf.zeros([tf.shape(X)[0], tf.shape(X2)[0]], dtype=tf.float64)
        return tf.linalg.diag(self.K_diag(X))

    def K_diag(self, X, presliced=False):
        idx = tf.cast(X[:, 0], tf.int32)
        noises = tf.gather(self.noise, idx)
        return noises

#from .multikernel import MultiKernel
#
#class Noise(MultiKernel):
#    def __init__(self, input_dim, output_dim, active_dim=None):
#        """
#        - input_dim (int): The number of input dimensions.
#        - output_dim (int): The number of output dimensions.
#        - active_dims (list of int): Apply kernel to specified dimensions only.
#        """
#
#        noise = np.random.random((output_dim))
#
#        MultiKernel.__init__(self, input_dim, output_dim, active_dim)
#        self.noise = gpflow.Parameter(noise, transform=gpflow.utilities.positive(), name="noise")
#
#    def subK(self, index, X, X2):
#        i, j = index
#        if i != j or not np.array_equal(X, X2):
#            K = tf.zeros([tf.shape(X)[0], tf.shape(X2)[0]], dtype=tf.float64)
#        else:
#            K = tf.linalg.diag(tf.fill([tf.shape(X)[0]], self.noise[i]))
#        return K

