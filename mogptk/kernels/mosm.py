import gpflow
from gpflow.params import Parameter
from gpflow import transforms
import numpy as np
import tensorflow as tf
from tensorflow import reduce_sum as rsum
from tensorflow import reduce_prod as rprod

from .fixphase import FixPhase
from .fixdelay import FixDelay
from .multikernel import MultiKernel

# TODO: dont use angular frequency
class MultiOutputSpectralMixture(MultiKernel):
    def __init__(
        self,
        input_dim,
        output_dim,
        magnitude=None,
        mean=None,
        variance=None,
        delay=None,
        phase=None,
        active_dim=None,
        magnitude_prior=None):
        """
        - input_dim (int) is the input dimension
        - output_dim (int) is the output dimension
        - magnitude (np.ndarray) has shape (output_dim)
        - mean (np.ndarray) has shape (input_dim, output_dim)
        - variance (np.ndarray) has shape (input_dim, output_dim)
        - delay (np.ndarray) has shape (input_dim, output_dim)
        - phase (np.ndarray) has shape (output_dim)
        """

        if magnitude is None:
            magnitude = np.random.standard_normal((output_dim))
        if mean is None:
            mean = np.random.standard_normal((input_dim, output_dim))
        if variance is None:
            variance = np.random.random((input_dim, output_dim))
        if delay is None:
            delay = np.zeros((input_dim, output_dim))
        if phase is None:
            phase = np.zeros((output_dim))

        if magnitude.shape != (output_dim,):
            raise Exception("bad magnitude shape %s" % (magnitude.shape,))
        if mean.shape != (input_dim, output_dim):
            raise Exception("bad mean shape %s" % (mean.shape,))
        if variance.shape != (input_dim, output_dim):
            raise Exception("bad variance shape %s" % (variance.shape,))
        if delay.shape != (input_dim, output_dim):
            raise Exception("bad delay shape %s" % (delay.shape,))
        if phase.shape != (output_dim,):
            raise Exception("bad phase shape %s" % (phase.shape,))

        MultiKernel.__init__(self, input_dim, output_dim, active_dim)
        self.magnitude = Parameter(magnitude, prior=magnitude_prior)
        self.mean = Parameter(mean, transform=transforms.positive)
        self.variance = Parameter(variance, transform=transforms.positive)
        self.delay = Parameter(delay)
        self.phase = Parameter(phase)

    def subK(self, index, X, X2):
        i, j = index
        if i == j:
            mean = tf.expand_dims(tf.slice(self.mean, [0, i], [self.input_dim, 1]), axis=2)
            temp = np.power(2 * np.pi, self.input_dim / 2) \
                    * tf.sqrt(rprod(self.variance[:, i])) \
                    * tf.square(self.magnitude[i])
            K = temp * tf.exp(-0.5 * self.sqdist(X, X, self.variance[:, i])) \
                    * tf.cos(rsum(mean * self.dist(X, X), 0))
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
                    * tf.exp(-0.25 * rsum(tf.square(self.mean[:, i] - self.mean[:, j]) / sv))

            alpha = np.power(2 * np.pi, self.input_dim / 2) * tf.sqrt(rprod(cross_var)) * cross_magnitude
            K = alpha * tf.exp(-0.5 * self.sqdist(X + self.delay[:, i], X2 + self.delay[:, j], cross_var)) \
                    * tf.cos(rsum(cross_mean * (self.dist(X, X2) + cross_delay), axis=0) + cross_phase)
        return K
