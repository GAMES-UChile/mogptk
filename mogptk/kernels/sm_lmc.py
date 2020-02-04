import numpy as np
import gpflow
import tensorflow as tf

from .multikernel import MultiKernel

# uses angular freq
class SpectralMixtureLMC(MultiKernel):
    def __init__(self, input_dim, output_dim, Rq, active_dims=None, name='sm_lmc'):
        """
        - input_dim (int): The number of input dimensions.
        - output_dim (int): The number of output dimensions.
        - Rq (int): The number of subcomponents.
        - active_dims (list of int): Apply kernel to specified dimensions only.
        """

        constant = np.random.standard_normal((Rq, output_dim))
        mean = np.random.random((input_dim))
        variance = np.random.random((input_dim))

        MultiKernel.__init__(self, input_dim, output_dim, active_dims, name=name)
        self.constant = gpflow.Parameter(constant, name="constant")
        self.mean = gpflow.Parameter(mean, transform=gpflow.utilities.positive(), name="mean")
        self.variance = gpflow.Parameter(variance, transform=gpflow.utilities.positive(), name="variance")

    def subK(self, index, X, X2):
        i, j = index
        Tau = self.dist(X, X2)
        constants = self.constant[:,i] * self.constant[:,j]
        exp_term = tf.square(Tau) * tf.expand_dims(tf.expand_dims(self.variance, 1), 2)
        exp_term = (-1/2)*tf.reduce_sum(exp_term, axis=0)
        exp = tf.exp(exp_term)
        constants_times_exp = tf.expand_dims(tf.expand_dims(constants, 1), 2)*exp
        cos_term = tf.reduce_sum(tf.expand_dims(tf.expand_dims(self.mean, 1), 2) * Tau, axis=0)
        cos = tf.cos(cos_term)
        return tf.reduce_sum(constants_times_exp * cos, axis=0)
