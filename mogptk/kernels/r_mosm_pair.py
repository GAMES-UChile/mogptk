import gpflow
import numpy as np
import tensorflow as tf

from .multikernel import MultiKernel

# TODO: dont use angular frequency
class RestrictedMultiOutputSpectralMixture_p(MultiKernel):
    def __init__(self, input_dim, output_dim, active_dim=None, magnitude_prior=None, name='rmosm_p', channels=None):
        """
        - input_dim (int): The number of input dimensions.
        - output_dim (int): The number of output dimensions.
        - active_dims (list of int): Apply kernel to specified dimensions only.
        - magnitude_prior (gpflow.Prior): Prior on the magnitude.
        """

        magnitude = np.random.standard_normal((2))
        mean = np.random.random((input_dim, 2))
        variance = np.random.random((input_dim, 2))
        delay = np.zeros((input_dim, 2))
        phase = np.zeros((2))

        MultiKernel.__init__(self, input_dim, output_dim, active_dim, name=name)
        
        self.magnitude = gpflow.Parameter(magnitude, prior=magnitude_prior, name="magnitude")
        
        self.mean = gpflow.Parameter(mean, transform=gpflow.utilities.positive(), name="mean")
        self.variance = gpflow.Parameter(variance, transform=gpflow.utilities.positive(), name="variance")
        if output_dim != 1:
            self.delay = gpflow.Parameter(delay, name="delay")
            self.phase = gpflow.Parameter(phase, name="phase")

        self.channels = channels

    def subK(self, index, X, X2):
        i, j = index

        if (i == j == self.channels[0]) or (i == j == self.channels[1]):

            if i == self.channels[0]:
                i = 0
            else:
                i = 1

            mean = tf.expand_dims(tf.slice(self.mean, [0, i], [self.input_dim, 1]), axis=2)
            temp = np.power(2 * np.pi, self.input_dim / 2) \
                    * tf.sqrt(tf.reduce_prod(self.variance[:, i])) \
                    * tf.square(self.magnitude[i])
            K = temp * tf.exp(-0.5 * self.sqdist(X, X2, self.variance[:, i])) \
                    * tf.cos(tf.reduce_sum(mean * self.dist(X, X2), 0))

        elif (i==self.channels[0] and j==self.channels[1]) or (j==self.channels[0] and i==self.channels[1]):

            if (i==self.channels[0] and j==self.channels[1]):
                i = 0
                j = 1
            else:
                i = 1
                j = 0

            sv = self.variance[:, i] + self.variance[:, j]
            cross_delay = tf.reshape(
                self.delay[:, i] - self.delay[:, j],
                [self.input_dim, 1, 1]
            )
            cross_phase = self.phase[i] - self.phase[j]
            cross_var = (2 * self.variance[:, i] * self.variance[:, j]) / sv
            cross_mean = tf.reshape(
                (self.variance[:, i] * self.mean[:, j] + self.variance[:, j] * self.mean[:, i]) / sv,
                [self.input_dim, 1, 1]
            )
            cross_magnitude = self.magnitude[i] * self.magnitude[j] \
                    * tf.exp(-0.25 * tf.reduce_sum(tf.square(self.mean[:, i] - self.mean[:, j]) / sv))

            alpha = np.power(2 * np.pi, self.input_dim / 2) * tf.sqrt(tf.reduce_prod(cross_var)) * cross_magnitude
            K = alpha * tf.exp(-0.5 * self.sqdist(X + self.delay[:, i], X2 + self.delay[:, j], cross_var)) \
                    * tf.cos(tf.reduce_sum(cross_mean * (self.dist(X, X2) + cross_delay), axis=0) + cross_phase)
        else:
            K = tf.zeros([tf.shape(X)[0], tf.shape(X2)[0]], dtype=tf.float64)
        return K
