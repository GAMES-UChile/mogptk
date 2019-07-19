import tensorflow as tf
import numpy as np
from gpflow.decors import params_as_tensors
from gpflow.params import Parameter
from gpflow.kernels import Kernel,RBF,Cosine
from gpflow import transforms, autoflow,settings

class SpectralMixtureOLD(Kernel):
    def __init__(self, num_mixtures=1, mixture_weights=None, mixture_scales=None,mixture_means=None, input_dim=1, active_dims=None, name=None):
        super().__init__(input_dim, active_dims, name=name)
        self.num_mixtures = Parameter(num_mixtures, trainable=False)
        self.mixture_weights = Parameter(mixture_weights)
        self.mixture_scales = Parameter(mixture_scales,transform=transforms.positive)
        self.mixture_means = Parameter(mixture_means)

    @params_as_tensors
    def K(self, X1, X2=None):
        if X2 is None:
            X2 = X1
        X1 = tf.transpose(tf.expand_dims(X1, -1), perm=[1, 0, 2])
        X2 = tf.expand_dims(tf.transpose(X2, perm=[1, 0]), -2)

        Tau = tf.subtract(X1, X2)

        cos_term = tf.tensordot(self.mixture_means, Tau, axes=((1),(0)))

        scales_expand = tf.expand_dims(tf.expand_dims(tf.transpose(self.mixture_scales), -2), -2)

        Tau_tile = tf.tile(tf.expand_dims(Tau,-1),(1,1,1,self.num_mixtures))

        exp_term_arg = tf.multiply(tf.square(Tau_tile), scales_expand)
        exp_term = tf.multiply(tf.transpose(tf.reduce_sum(exp_term_arg, 0),perm=[2, 0, 1]), -0.5)

        squared_weights = np.power(2*np.pi, self.input_dim/2)*np.power(self.mixture_weights,2)
        det_mixture_scales = tf.reduce_prod(self.mixture_scales, 1)

        squared_weights = tf.multiply(squared_weights, tf.sqrt(det_mixture_scales))
        weights = tf.expand_dims(tf.expand_dims(squared_weights,-1),-1)
        weights = tf.tile(weights,(1,tf.shape(X1)[1],tf.shape(X2)[2]))

        complete_expression = tf.reduce_sum(tf.multiply(weights,tf.multiply(tf.exp(exp_term),tf.cos(cos_term))),0)
        return complete_expression

    @params_as_tensors
    def Kdiag(self, X):
        return tf.diag_part(self.K(X,X))

def sm_unravel_hp(hp, num_mixtures):
    means = []
    for x in hp[:num_mixtures]:
        means.append(np.array(x))
    means = np.array(means).reshape(-1,1)

    scales = []
    for x in hp[num_mixtures:num_mixtures*2]:
        scales.append(x)
    scales = np.array([scales])

    weights = []
    for x in hp[num_mixtures*2:num_mixtures*3]:
        weights.append(x)
    weights = np.array(weights)

    return weights, means, scales
