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
    def __init__(self, input_dim, output_dim, active_dims=None):
        """
        - input_dim (int): The number of input dimensions.
        - output_dim (int): The number of output dimensions.
        - active_dims (list of int): Apply kernel to specified dimensions only.
        """

        constant = np.random.random((output_dim))
        variance = np.zeros((input_dim, output_dim))
        latent_variance = np.random.random((input_dim))

        MultiKernel.__init__(self, input_dim, output_dim, active_dims)
        self.constant = Parameter(constant, transform=transforms.positive)
        self.variance = Parameter(variance, transform=transforms.positive)
        self.latent_variance = Parameter(latent_variance, transform=transforms.positive)

    def subK(self, index, X, X2):
        i, j = index
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

