from gpflow.params import Parameter
from gpflow import transforms
import numpy as np
import tensorflow as tf
from tensorflow import reduce_sum as rsum
from tensorflow import reduce_prod as rprod

from .fixphase import FixPhase
from .fixdelay import FixDelay
from .multikernel import MultiKern

# from gpflow._settings import settings
# float_type = settings.dtypes.float_type


class MultiSpectralMixture(MultiKern):
    def __init__(self, input_dim, output_dim,
                 spectral_constant=None, spectral_mean=None,
                 spectral_variance=None, spectral_delay=None,
                 spectral_phase=None, active_dims=None):
        """
        - input_dim is the input dimension as integer
        - output_dim is the output Dimension as integer
        - spectral_constant is a tensor of rank 1 of length output_dim
        - spectral_mean is a tensor of rank 2 of shape
          (input_dim, output_dim)
        - spectral_variance is a tensor of rank 2 of shape
          (input_dim, output_dim)
        - spectral_delay is a tensor of rank 2 of shape
          (input_dim, output_dim)
        - spectral_phase is a tensor of rank 1 of length output_dim

        TODO: ADD AUTOMATIC INITIALIZATION
        """
        if spectral_constant is None:
            spectral_constant = np.random.randn(output_dim)
        if spectral_mean is None:
            spectral_mean = np.ones([input_dim, output_dim])
        if spectral_variance is None:
            spectral_variance = np.ones([input_dim, output_dim])
        if spectral_delay is None:
            spectral_delay = np.zeros([input_dim, output_dim])
        if spectral_phase is None:
            spectral_phase = np.zeros(output_dim)

        MultiKern.__init__(self, input_dim, output_dim, active_dims)
        self.constant = Parameter(spectral_constant)
        self.mean = Parameter(spectral_mean)
        self.variance = Parameter(spectral_variance, transforms.positive)
        self.delay = Parameter(spectral_delay, FixDelay(input_dim, output_dim))
        self.phase = Parameter(spectral_phase, FixPhase())
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
                mean = tf.expand_dims(tf.slice(self.mean, [0, i],
                                               [self.input_dim, 1]),
                                      axis=2)
                temp = np.power(2 * np.pi, self.input_dim / 2) \
                    * tf.sqrt(rprod(self.variance[:, i])) \
                    * tf.square(self.constant[i])
                return temp * tf.exp(-0.5 * self.sqdist(X, X2, self.variance[:, i])) \
                    * tf.cos(rsum(mean * self.dist(X, X2), 0))
        else:
            def cov_function(X, X2):
                sv = self.variance[:, i] + self.variance[:, j]
                cross_delay = tf.reshape(self.delay[:, i] - self.delay[:, j],
                                         [self.input_dim, 1, 1])
                cross_phase = self.phase[i] - self.phase[j]
                cross_var = (2 * self.variance[:, i] * self.variance[:, j]) / sv
                cross_mean = tf.reshape(
                    (self.variance[:, i] * self.mean[:, j] + self.variance[:, j] * self.mean[:, i]) / sv,
                    [self.input_dim, 1, 1]
                )
                cross_magnitude = self.constant[i] * self.constant[j] \
                    * tf.exp(-0.25 * rsum(tf.square(self.mean[:, i] - self.mean[:, j]) / sv))
                alpha = np.power(2 * np.pi, self.input_dim / 2) * tf.sqrt(rprod(cross_var)) * cross_magnitude
                return alpha * tf.exp(-0.5 * self.sqdist(X + self.delay[:, i], X2 + self.delay[:, j], cross_var)) \
                    * tf.cos(rsum(cross_mean * (self.dist(X, X2) + cross_delay), axis=0) + cross_phase)
        return cov_function

    def sqdist(self, X, X2, lscales):
        """Return the square distance between two tensors."""
        Xs = tf.reduce_sum(tf.square(X) * lscales, 1)
        if X2 is None:
            return -2 * tf.matmul(X * lscales, tf.transpose(X)) \
                + tf.reshape(Xs, (-1, 1)) + tf.reshape(Xs, (1, -1))
        else:
            X2s = tf.reduce_sum(tf.square(X2) * lscales, 1)
            return -2 * tf.matmul(X * lscales, tf.transpose(X2)) \
                + tf.reshape(Xs, (-1, 1)) + tf.reshape(X2s, (1, -1))

    def dist(self, X, X2):
        """Return the distance between two tensors (?)."""
        if X2 is None:
            X2 = X
        X = tf.expand_dims(tf.transpose(X), axis=2)
        X2 = tf.expand_dims(tf.transpose(X2), axis=1)
        return tf.matmul(X, tf.ones_like(X2)) + tf.matmul(tf.ones_like(X), -X2)
