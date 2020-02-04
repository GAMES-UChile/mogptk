import numpy as np
import gpflow
import tensorflow as tf

from .multikernel import MultiKernel

class ConvolutionalGaussian(MultiKernel):
    def __init__(self, input_dim, output_dim, active_dims=None, name='conv'):
        """
        - input_dim (int): The number of input dimensions.
        - output_dim (int): The number of output dimensions.
        - active_dims (list of int): Apply kernel to specified dimensions only.
        """

        constant = np.random.random((output_dim))
        variance = np.ones((input_dim, output_dim)) * 10

        MultiKernel.__init__(self, input_dim, output_dim, active_dims, name=name)
        self.constant = gpflow.Parameter(constant, transform=gpflow.utilities.positive(), name="constant")
        self.variance = gpflow.Parameter(variance, transform=gpflow.utilities.positive(), name="variance")

    def subK(self, index, X, X2):
        i, j = index
        Tau = self.dist(X,X2)
        constants = tf.expand_dims(tf.expand_dims(tf.expand_dims(self.constant[i]*self.constant[j],0),1),2)

        if i == j:
            complete_variance = self.variance[:,i]
        else:
            complete_variance = 2 * self.variance[:,i] * self.variance[:,j] / (self.variance[:,i] + self.variance[:,j])

        exp_term = tf.square(Tau)*tf.expand_dims(tf.expand_dims(complete_variance,1),2)
        exp_term = (-1/2)*tf.reduce_sum(exp_term, axis = 0)
        exp = tf.exp(exp_term)
        return tf.reduce_sum(constants*exp, axis=0)

