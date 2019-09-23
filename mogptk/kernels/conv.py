from gpflow.params import Parameter
from gpflow import transforms
import numpy as np
import tensorflow as tf
from tensorflow import reduce_sum as rsum
from tensorflow import reduce_prod as rprod

from .fixphase import FixPhase
from .fixdelay import FixDelay
from .multikernel import MultiKernel

class ConvolutionalGaussian(MultiKernel):
    def __init__(self, input_dim, output_dim, constant=None, variance=None, latent_variance=None, active_dims=None):
        """
        - input_dim (int) is the input dimension
        - output_dim (int) is the output dimension
        - constant (np.ndarray) has shape (output_dim)
        - variance (np.ndarray) has shape (input_dim, output_dim)
        """

        if constant is None:
            constant = np.random.random((output_dim))
        if variance is None:
            variance = np.random.random((input_dim, output_dim))
        if latent_variance is None:
            latent_variance = np.random.random((input_dim))

        if constant.shape != (output_dim,):
            raise Exception("bad constant shape %s" % (constant.shape,))
        if variance.shape != (input_dim, output_dim):
            raise Exception("bad variance shape %s" % (variance.shape,))
        if latent_variance.shape != (input_dim,):
            raise Exception("bad latent variance shape %s" % (latent_variance.shape,))

        MultiKernel.__init__(self, input_dim, output_dim, active_dims)
        self.constant = Parameter(constant, transform=transforms.positive)
        self.variance = Parameter(variance, transform=transforms.positive)
        self.latent_variance = Parameter(latent_variance, transform=transforms.positive)
        self.kerns = [[self._kernel_factory(i, j) for j in range(output_dim)] for i in range(output_dim)]

    def subK(self, index, X, X2=None):
        i, j = index
        return self.kerns[i][j](X, X2)

    def subKdiag(self, index, X):
        K = self.subK((index, index), X, X)
        return tf.diag_part(K)

    def _kernel_factory(self, i, j):
        """Return a function that calculates proper sub-kernel."""
        def cov_function(X,X2):
            Tau = self.dist(X,X2)
            # constants = tf.expand_dims(tf.expand_dims(tf.expand_dims(self.constant[i]*self.constant[j],0),1),2)

            # cross_var = self.variance[:,i] + self.variance[:,j]
            # if i == j:
            #    cross_var = self.variance[:,i] + 1e-8
            #else:
            #    cross_var = (self.variance[:,i] * self.variance[:,i]) / (self.variance[:,i] + self.variance[:,j])
            cross_var = 1 / (1 / self.variance[:, i] + 1 / self.variance[:, j] + 1 / self.latent_variance)

            cross_magnitude = self.constant[i] * self.constant[j] * tf.sqrt(1 / self.latent_variance) * tf.sqrt(cross_var)

            if i == j:
                cross_var = cross_var + 1e-8

            # exp_term = tf.square(Tau)*tf.expand_dims(tf.expand_dims(cross_var,1),2)
            # exp_term = (-1/2)*tf.reduce_sum(exp_term, axis = 0)
            # exp = tf.exp(exp_term)
            # complete_expression = tf.reduce_sum(cross_magnitude*exp, axis=0, name="cov_function")
            # return complete_expression
            return cross_magnitude * tf.exp(-0.5 * self.sqdist(X, X2, cross_var))
        return cov_function

    def dist(self, X, X2):
        if X2 is None:
            X2 = X
        X = tf.expand_dims(tf.transpose(X), axis=2)
        X2 = tf.expand_dims(tf.transpose(X2), axis=1)
        return tf.matmul(X, tf.ones_like(X2)) + tf.matmul(tf.ones_like(X), -X2)

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
