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
        self.kerns = [[self._kernel_factory(i, j) for j in range(output_dim)] for i in range(output_dim)]

    def subK(self, index, X, X2=None):
        i, j = index
        return self.kerns[i][j](X, X2)

    def subKdiag(self, index, X):
        K = self.subK((index, index), X, X)
        return tf.diag_part(K)

    def _kernel_factory(self, i, j):
        """Return a function that calculates proper sub-kernel."""
        if i == j:
            def cov_function(X, X2):
                if X==X2:
                    d = tf.fill(tf.shape(X)[:-1], self.noise[i])
                    return tf.cast(tf.matrix_diag(d), tf.float64)
                else:
                    shape = tf.concat([tf.shape(X)[:-1], tf.shape(X2)[:-1]], 0)
                    return tf.cast(tf.fill(shape, 0.0), tf.float64)
        else:
            def cov_function(X, X2):
                shape = tf.concat([tf.shape(X)[:-1], tf.shape(X2)[:-1]], 0)
                return tf.cast(tf.fill(shape, 0.0), tf.float64)
        return cov_function


    def dist(self, X, X2):
        if X2 is None:
            X2 = X
        X = tf.expand_dims(tf.transpose(X), axis=2)
        X2 = tf.expand_dims(tf.transpose(X2), axis=1)
        return tf.matmul(X, tf.ones_like(X2)) + tf.matmul(tf.ones_like(X), -X2)
