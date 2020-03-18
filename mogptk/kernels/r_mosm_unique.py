import gpflow
import numpy as np
import tensorflow as tf

from .multikernel import MultiKernel

# TODO: dont use angular frequency
class RestrictedMultiOutputSpectralMixture_u(MultiKernel):
    def __init__(self, input_dim, output_dim, active_dim=None, magnitude_prior=None, name='rmosm_u', channels=None):
        """
        - input_dim (int): The number of input dimensions.
        - output_dim (int): The number of output dimensions.
        - active_dims (list of int): Apply kernel to specified dimensions only.
        - magnitude_prior (gpflow.Prior): Prior on the magnitude.
        """

        magnitude = np.random.standard_normal((1))
        mean = np.random.random((input_dim, 1))
        variance = np.random.random((input_dim, 1))
        delay = np.zeros((input_dim, 1))
        phase = np.zeros((1))

        MultiKernel.__init__(self, input_dim, output_dim, active_dim, name=name)
        self.magnitude = gpflow.Parameter(magnitude, prior=magnitude_prior, name="magnitude")
        self.mean = gpflow.Parameter(mean, transform=gpflow.utilities.positive(), name="mean")
        self.variance = gpflow.Parameter(variance, transform=gpflow.utilities.positive(), name="variance")

        self.channels = channels

    def subK(self, index, X, X2):
        i, j = index    
        if i == j == self.channels[0]:
            mean = tf.expand_dims(tf.slice(self.mean, [0, 0], [self.input_dim, 1]), axis=2)
            temp = np.power(2 * np.pi, self.input_dim / 2) \
                    * tf.sqrt(tf.reduce_prod(self.variance[:])) \
                    * tf.square(self.magnitude)
            K = temp * tf.exp(-0.5 * self.sqdist(X, X2, self.variance[:])) \
                    * tf.cos(tf.reduce_sum(mean * self.dist(X, X2), 0))
        else:
            K = tf.zeros([tf.shape(X)[0], tf.shape(X2)[0]], dtype=tf.float64)

        return K
