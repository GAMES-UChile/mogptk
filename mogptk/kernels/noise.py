import gpflow
import numpy as np
import tensorflow as tf

from .multikernel import MultiKernel

class Noise(MultiKernel):
    def __init__(self, input_dim, output_dim, active_dim=None):
        """
        - input_dim (int): The number of input dimensions.
        - output_dim (int): The number of output dimensions.
        - active_dims (list of int): Apply kernel to specified dimensions only.
        """

        noise = np.random.random((output_dim))

        MultiKernel.__init__(self, input_dim, output_dim, active_dim)
        self.noise = gpflow.Parameter(noise, transform=gpflow.utilities.positive())

    def subK(self, index, X, X2):
        i, j = index
        if i != j or not np.array_equal(X, X2):
            K = tf.zeros([tf.shape(X)[0], tf.shape(X2)[0]], dtype=tf.float64)
        else:
            K = tf.linalg.diag(tf.fill([tf.shape(X)[0]], self.noise[i]))
        return K

