from gpflow.params import Parameter
from gpflow import transforms
import numpy as np
import tensorflow as tf
from tensorflow import reduce_sum as rsum
from tensorflow import reduce_prod as rprod

from .fixphase import FixPhase
from .fixdelay import FixDelay
from .multikernel import MultiKernel

# TODO: swap input/output dimension order, so that the first variable dim is always output_dims, and if it has a second dim it is input_dims
class MultiOutputSpectralMixture(MultiKernel):
    def __init__(self, input_dim, output_dim, magnitude=None, mean=None, variance=None, delay=None, phase=None, noise=None, active_dim=None):
        """
        - input_dim (int) is the input dimension
        - output_dim (int) is the output dimension
        - magnitude (np.ndarray) has shape (output_dim)
        - mean (np.ndarray) has shape (input_dim, output_dim)
        - variance (np.ndarray) has shape (input_dim, output_dim)
        - delay (np.ndarray) has shape (input_dim, output_dim)
        - phase (np.ndarray) has shape (output_dim)
        - noise (np.ndarray) has shape (output_dim)
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
        if noise is None:
            noise = np.random.random((output_dim))

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
        if noise.shape != (output_dim,):
            raise Exception("bad noise shape %s" % (noise.shape,))

        MultiKernel.__init__(self, input_dim, output_dim, active_dim)
        self.magnitude = Parameter(magnitude)
        self.mean = Parameter(mean)
        self.variance = Parameter(variance, transform=transforms.positive)
        self.delay = Parameter(delay, FixDelay(input_dim, output_dim))
        self.phase = Parameter(phase, FixPhase())
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
                mean = tf.expand_dims(tf.slice(self.mean, [0, i], [self.input_dim, 1]), axis=2)
                temp = np.power(2 * np.pi, self.input_dim / 2) \
                        * tf.sqrt(rprod(self.variance[:, i])) \
                        * tf.square(self.magnitude[i])
                return temp * tf.exp(-0.5 * self.sqdist(X, X2, self.variance[:, i])) \
                        * tf.cos(rsum(mean * self.dist(X, X2), 0)) + self.noise[i]
        else:
            def cov_function(X, X2):
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
                return alpha * tf.exp(-0.5 * self.sqdist(X + self.delay[:, i], X2 + self.delay[:, j], cross_var)) \
                        * tf.cos(rsum(cross_mean * (self.dist(X, X2) + cross_delay), axis=0) + cross_phase)
        return cov_function


    #* performs element-wise multiplication
    #-1 means infer shape for that dimension
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

    # def sqdist(self, X, X2, lscales):
    #     """Return the square distance between two tensors."""
    #     Xs = tf.reduce_sum(tf.square(X) * lscales, axis = -1, keepdims=True)
    #     if X2 is None:
    #         return -2 * tf.matmul(X * lscales, X2, transpose_b=True) + Xs + tf.matrix_transpose(Xs)
    #     else:
    #         X2s = tf.reduce_sum(tf.square(X2) * lscales, axis = -1, keepdims=True)
    #         return -2 * tf.matmul(X * lscales, X2, transpose_b=True) + Xs + tf.matrix_transpose(X2s)

    def dist(self, X, X2):
        if X2 is None:
            X2 = X
        X = tf.expand_dims(tf.transpose(X), axis=2)
        X2 = tf.expand_dims(tf.transpose(X2), axis=1)
        return tf.matmul(X, tf.ones_like(X2)) + tf.matmul(tf.ones_like(X), -X2)
