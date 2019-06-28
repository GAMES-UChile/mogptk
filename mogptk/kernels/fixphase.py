import gpflow
from gpflow.transforms import Transform
import numpy as np
import tensorflow as tf


class FixPhase(Transform):
    """A transform that prevents a phase parameter from being updated during training."""
    def __init__(self):
        gpflow.transforms.Transform.__init__(self)
        self.fixed_inds = np.array([0])
        self.fixed_vals = np.zeros([1])

    def forward(self, x):
        total_size = x.shape[0] + self.fixed_inds.shape[0]
        nonfixed_inds = np.setdiff1d(np.arange(total_size), self.fixed_inds) # TODO: deprecated
        y = np.empty(total_size)
        y[nonfixed_inds] = x
        y[self.fixed_inds] = self.fixed_vals
        return y

    def backward(self, y):
        nonfixed_inds = np.setdiff1d(np.arange(y.shape[0]), self.fixed_inds) # TODO: deprecated
        x = y[nonfixed_inds]
        return x

    def forward_tensor(self, x):
        total_size = tf.shape(x)[0] + self.fixed_inds.shape[0]
        nonfixed_inds = tf.setdiff1d(tf.range(total_size), self.fixed_inds)[0] # TODO: deprecated
        y = tf.dynamic_stitch([self.fixed_inds, nonfixed_inds],
                              [self.fixed_vals, x])
        return y
    #New function
    def backward_tensor(self, y):
        nonfixed_inds = tf.setdiff1d(tf.range(y.shape[0]), self.fixed_inds) # TODO: deprecated
        x = y[nonfixed_inds]
        return x

    def log_jacobian_tensor(self, x):
        return 0.0

    def __str__(self):
        return 'PartiallyFixed'
