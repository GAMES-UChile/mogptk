from gpflow.params import Parameter
from gpflow import transforms
import numpy as np
import tensorflow as tf
from tensorflow import reduce_sum as rsum
from tensorflow import reduce_prod as rprod

from .fixphase import FixPhase
from .fixdelay import FixDelay
from .multikernel import MultiKernel

# uses angular freq
class SpectralMixtureLMC(MultiKernel):
    def __init__(self, input_dim, output_dim, Rq, constant=None, mean=None, variance=None, active_dims=None):
        """
        - input_dim (int) is the input dimension
        - output_dim (int) is the output dimension
        - constant (np.ndarray) has shape (Rq, output_dim)
        - mean (np.ndarray) has shape (input_dim)
        - variance (np.ndarray) has shape (input_dim)
        """

        if constant is None:
            constant = np.random.standard_normal((Rq, output_dim))
        if mean is None:
            mean = np.random.random((input_dim))
        if variance is None:
            variance = np.random.random((input_dim))

        if constant.shape != (Rq, output_dim):
            raise Exception("bad constant shape %s" % (constant.shape,))
        if mean.shape != (input_dim,):
            raise Exception("bad mean shape %s" % (mean.shape,))
        if variance.shape != (input_dim,):
            raise Exception("bad variance shape %s" % (variance.shape,))

        MultiKernel.__init__(self, input_dim, output_dim, active_dims)
        self.constant = Parameter(constant)
        self.mean = Parameter(mean, transform = transforms.positive)
        self.variance = Parameter(variance, transform=transforms.positive)

    def subK(self, index, X, X2):
        i, j = index
        Tau = self.dist(X,X2)
        constants = self.constant[:,i]*self.constant[:,j]
        exp_term = tf.square(Tau)*tf.expand_dims(tf.expand_dims(self.variance,1),2)
        exp_term = (-1/2)*tf.reduce_sum(exp_term, axis = 0)
        exp = tf.exp(exp_term)
        constants_times_exp = tf.expand_dims(tf.expand_dims(constants,1),2)*exp
        cos_term = tf.reduce_sum(tf.expand_dims(tf.expand_dims(self.mean,1),2)*Tau, axis=0)
        cos = tf.cos(cos_term)
        complete_expression = tf.reduce_sum(constants_times_exp*cos, axis = 0)
        return complete_expression
