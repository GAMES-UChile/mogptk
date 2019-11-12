from gpflow.params import Parameter
from gpflow import transforms
import numpy as np
import tensorflow as tf

from .multikernel import MultiKernel

class Noise(MultiKernel):
    def __init__(self, input_dim, output_dim, noise=None, active_dim=None):
        """
        - input_dim (int) is the input dimension
        - output_dim (int) is the output dimension
        - noise (np.ndarray) has shape (output_dim)
        """

        if noise is None:
            noise = np.random.random((output_dim))

        if noise.shape != (output_dim,):
            raise Exception("bad noise shape %s" % (noise.shape,))

        MultiKernel.__init__(self, input_dim, output_dim, active_dim)
        self.noise = Parameter(noise, transform=transforms.positive)

    def subK(self, index, X, X2=None):
        shape = tf.concat([tf.shape(X)[:-1], tf.shape(X2)[:-1]], 0)
        K = tf.cast(tf.fill(shape, 0.0), tf.float64)
        return K

    def subKdiag(self, index, X):
        i = index
        d = tf.fill(tf.shape(X)[:-1], self.noise[i])
        K = tf.cast(tf.matrix_diag(d), tf.float64)
        return tf.diag_part(K)

