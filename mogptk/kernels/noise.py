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

    def subK(self, index, X, X2):
        i, j = index
        if i != j or X != X2:
            K = tf.zeros([tf.shape(X)[0], tf.shape(X2)[0]], dtype=tf.float64)
        else:
            K = tf.matrix_diag(tf.fill([tf.shape(X)[0]], self.noise[i]))
        return K

