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


class ConvolutionalGaussian(MultiKern):
    def __init__(self, input_dim, output_dim, spectral_constant=None, spectral_component_variance=None, spectral_channel_variance = None, active_dims=None):

        if spectral_constant is None:
            spectral_constant = np.random.random(output_dim)
            # spectral_constant = np.random.randn(output_dim)
        if spectral_component_variance is None:
            spectral_component_variance = np.random.random(input_dim)
        if spectral_channel_variance is None:
            # spectral_channel_variance = np.random.random((input_dim, output_dim))
            spectral_channel_variance = np.zeros((input_dim, output_dim))
            # spectral_channel_variance = np.random.random((output_dim, input_dim))

        MultiKern.__init__(self, input_dim, output_dim, active_dims)
        self.constant = Parameter(spectral_constant, transform = transforms.positive)
        self.component_variance = Parameter(spectral_component_variance, transform = transforms.positive)
        self.channel_variance = Parameter(spectral_channel_variance, transform = transforms.positive)
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
            constants = tf.expand_dims(tf.expand_dims(tf.expand_dims(self.constant[i]*self.constant[j],0),1),2)
            complete_variance = self.channel_variance[:,i] + self.channel_variance[:,j] + self.component_variance
            # complete_variance = self.channel_variance[i,:] + self.channel_variance[j,:] + self.component_variance
            exp_term = tf.square(Tau)*tf.expand_dims(tf.expand_dims(complete_variance,1),2)
            exp_term = (-1/2)*tf.reduce_sum(exp_term, axis = 0)
            exp = tf.exp(exp_term)
            complete_expression = tf.reduce_sum(constants*exp, axis=0)
            return complete_expression
        return cov_function

    def dist(self, X, X2):
        if X2 is None:
            X2 = X
        X = tf.expand_dims(tf.transpose(X), axis=2)
        X2 = tf.expand_dims(tf.transpose(X2), axis=1)
        return tf.matmul(X, tf.ones_like(X2)) + tf.matmul(tf.ones_like(X), -X2)
