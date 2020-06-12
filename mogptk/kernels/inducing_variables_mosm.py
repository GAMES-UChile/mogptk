import numpy as np
import tensorflow as tf
from ..utils import splitback, reconstruct, sqdist, dist

BlockDiag = tf.linalg.LinearOperatorBlockDiag
Diag = tf.linalg.LinearOperatorDiag
LinearOp = tf.linalg.LinearOperatorFullMatrix

def Kuu_mosm_vik(inducing_variable, kernel, jitter=None):
    """
    Covariance between inducing variables 
    using variational inducing kernels.
    """
    Z, Z_index, K, Q = (lambda u: (u.z, u.z_index, u.K, u.Q))(inducing_variable)
    
    def Ker(x, q, inducing_variable):
        magnitude, variance, input_dim = (lambda u: (u.magnitude, u.variance, u.input_dim))(inducing_variable)
        temp = np.power(2 * np.pi, input_dim / 2) \
                * tf.sqrt(tf.reduce_prod(variance[:, q])) \
                * tf.square(magnitude[q])
        K = temp * tf.exp(-0.5 * sqdist(x, x, variance[:, q]))
        return K
    
    Zindex = tf.cast(Z_index, tf.int32)
    Zparts, Zsplitn, Zreturn = splitback(Z, Zindex, Q)
    # construct diag blocks
    block_list = []
    for q in range(Q):
        block_q = Ker(Zparts[q], q, inducing_variable)
        block_list.append(LinearOp(block_q))
    
    Kzz = BlockDiag(block_list).to_dense()
    return Kzz + jitter * tf.eye(len(inducing_variable), dtype=Kzz.dtype)


def Kuf_mosm_vik(inducing_variable, kernel, X):
    """
    Cross covariance between function values and inducing variables
    using variational inducing kernels.
    """
        
    def Ker(X, Z, i, q, inducing_variable, kernel):
        magnitude, variance = (lambda u: (u.magnitude, u.variance))(inducing_variable)

        assert inducing_variable.input_dim == kernel.input_dim
        input_dim = kernel.input_dim
        
        sv = kernel.variance[:, i] + variance[:, q]
        cross_delay = tf.reshape(
            kernel.delay[:, i],
            [input_dim, 1, 1]
        )
        cross_phase = kernel.phase[i]
        cross_var = (2 * kernel.variance[:, i] * variance[:, q]) / sv
        cross_mean = tf.reshape(
            variance[:, q] * kernel.mean[:, i] / sv,
            [input_dim, 1, 1]
        )
        cross_magnitude = kernel.magnitude[i] * magnitude[q] \
                * tf.exp(-0.25 * tf.reduce_sum(tf.square(kernel.mean[:, i]) / sv))

        alpha = np.power(2 * np.pi, input_dim / 2) * tf.sqrt(tf.reduce_prod(cross_var)) * cross_magnitude
        K = alpha * tf.exp(-0.5 * sqdist(X + kernel.delay[:, i], Z, cross_var)) \
                * tf.cos(tf.reduce_sum(cross_mean * (dist(X, Z) + cross_delay), axis=0) + cross_phase)
        return K

    Z, Z_index, K, Q = (lambda u: (u.z, u.z_index, u.K, u.Q))(inducing_variable)

    if hasattr(kernel, 'kernels'):
        output_dim = kernel.kernels[0].output_dim
    else:
        output_dim = kernel.output_dim

    Xindex = tf.cast(X[:, 0], tf.int32)  # find group indices
    Xparts, Xsplitn, Xreturn = splitback(X[:, 1:], Xindex, output_dim)
    
    Zindex = tf.cast(Z_index, tf.int32)
    Zparts, Zsplitn, Zreturn = splitback(Z, Zindex, Q)

    blocks = []
    for i in range(output_dim):
        row_i = []
        for q in range(Q):
            row_i.append(Ker(Xparts[i], Zparts[q], i, q, inducing_variable, kernel.kernels[q]))
        blocks.append(tf.concat(row_i, 1))
    Ksort = tf.concat(blocks, 0)

    Ktmp = reconstruct(Ksort, Xsplitn, Xreturn)
    KT = reconstruct(tf.transpose(Ktmp), Zsplitn, Zreturn)
    Kout = KT

    return Kout