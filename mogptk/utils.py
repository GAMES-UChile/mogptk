import tensorflow as tf

def splitback(data, indices, output_dim):
    """Apply dynamic_partitioning and calculate necessary statistics
    for inverse mapping."""
    # split data based on indices
    parts = tf.dynamic_partition(data, indices, output_dim)

    # to be able to invert the splitting, we need:
    # the size of each data split
    splitnum = tf.stack([tf.shape(x)[0] for x in parts])
    # indices to invert dynamic_part
    goback = tf.dynamic_partition(tf.range(tf.shape(data)[0]), indices, output_dim)
    return (parts, splitnum, goback)

def reconstruct(K, splitnum, goback):
    """Use quantities from splitback to invert a dynamic_partition."""
    tmp = tf.split(K, splitnum, axis=0)
    return tf.dynamic_stitch(goback, tmp)  # stitch

def sqdist(X, X2, lscales):
    """Return the square distance between two tensors."""
    Xs = tf.reduce_sum(tf.square(X) * lscales, 1)
    if X2 is None:
        return -2 * tf.matmul(X * lscales, tf.transpose(X)) \
                + tf.reshape(Xs, (-1, 1)) + tf.reshape(Xs, (1, -1))
    else:
        X2s = tf.reduce_sum(tf.square(X2) * lscales, 1)
        return -2 * tf.matmul(X * lscales, tf.transpose(X2)) \
                + tf.reshape(Xs, (-1, 1)) + tf.reshape(X2s, (1, -1))

def dist(X, X2):
    if X2 is None:
        X2 = X
    X = tf.expand_dims(tf.transpose(X), axis=2)
    X2 = tf.expand_dims(tf.transpose(X2), axis=1)
    return tf.matmul(X, tf.ones_like(X2)) + tf.matmul(tf.ones_like(X), -X2)


def set_trainable(model: tf.Module, flag: bool):
    """
    Set trainable flag for all `tf.Variable`s and `gpflow.Parameter`s in a module.
    """
    for variable in model.variables:
        variable._trainable = flag