import gpflow
from gpflow.transforms import Transform
import numpy as np
import tensorflow as tf

class FixDelay(Transform):
    """A transform that prevents a delay parameter from being updated during training."""
    def __init__(self, input_dim, groups):
        gpflow.transforms.Transform.__init__(self)
        self.fixed_inds = np.array([i * groups for i in range(input_dim)])
        self.fixed_vals = np.zeros(input_dim)

    def forward(self, x):
        y = x.reshape([-1])
        total_size = y.shape[0] + self.fixed_inds.shape[0]
        nonfixed_inds = np.setdiff1d(np.arange(total_size), self.fixed_inds)[0] # TODO: deprecated for tf.sets.difference()
        y = tf.reshape(tf.dynamic_stitch([self.fixed_inds, nonfixed_inds],
                                         [self.fixed_vals, y]),
                       [tf.shape(x)[0], -1])
        return y

    def backward(self, y):
        x = y.reshape([-1])
        nonfixed_inds = np.setdiff1d(np.arange(x.shape[0]), self.fixed_inds) # TODO: deprecated
        x = x[nonfixed_inds].reshape([y.shape[0], -1])
        return x

    def forward_tensor(self, x):
        y = tf.reshape(x, [-1])
        total_size = tf.shape(y)[0] + self.fixed_inds.shape[0]
        nonfixed_inds = tf.setdiff1d(tf.range(total_size), self.fixed_inds)[0] # TODO: deprecated
        y = tf.reshape(tf.dynamic_stitch([self.fixed_inds, nonfixed_inds],
                                         [self.fixed_vals, y]),
                       [tf.shape(x)[0], -1])
        return y
    #new function
    def backward_tensor(self, y):
        x = tf.reshape(y, [-1])
        nonfixed_inds = tf.setdiff1d(tf.range(x.shape[0]), self.fixed_inds) # TODO: deprecated
        x = x[nonfixed_inds].reshape([y.shape[0], -1])
        return x

    def log_jacobian_tensor(self, x):
        return 0.0

    def __str__(self):
        return 'PartiallyFixed'
