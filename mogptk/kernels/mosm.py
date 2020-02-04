import gpflow
import numpy as np
import tensorflow as tf

from .multikernel import MultiKernel

# TODO: dont use angular frequency
class MultiOutputSpectralMixture(MultiKernel):
    def __init__(self, input_dim, output_dim, active_dim=None, magnitude_prior=None, name='mosm'):
        """
        - input_dim (int): The number of input dimensions.
        - output_dim (int): The number of output dimensions.
        - active_dims (list of int): Apply kernel to specified dimensions only.
        - magnitude_prior (gpflow.Prior): Prior on the magnitude.
        """

        magnitude = np.random.standard_normal((output_dim))
        mean = np.random.random((input_dim, output_dim))
        variance = np.random.random((input_dim, output_dim))
        delay = np.zeros((input_dim, output_dim))
        phase = np.zeros((output_dim))

        MultiKernel.__init__(self, input_dim, output_dim, active_dim, name=name)
        self.magnitude = gpflow.Parameter(magnitude, prior=magnitude_prior, name="magnitude")
        self.mean = gpflow.Parameter(mean, transform=gpflow.utilities.positive(), name="mean")
        self.variance = gpflow.Parameter(variance, transform=gpflow.utilities.positive(), name="variance")
        if output_dim != 1:
            self.delay = gpflow.Parameter(delay, name="delay")
            self.phase = gpflow.Parameter(phase, name="phase")

    def subK(self, index, X, X2):
        i, j = index
        if i == j:
            mean = tf.expand_dims(tf.slice(self.mean, [0, i], [self.input_dim, 1]), axis=2)
            temp = np.power(2 * np.pi, self.input_dim / 2) \
                    * tf.sqrt(tf.reduce_prod(self.variance[:, i])) \
                    * tf.square(self.magnitude[i])
            K = temp * tf.exp(-0.5 * self.sqdist(X, X2, self.variance[:, i])) \
                    * tf.cos(tf.reduce_sum(mean * self.dist(X, X2), 0))
        else:
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
        return K
