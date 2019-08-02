# This is a simplification and adaptation to gpflow 1.0 of the work done
# by Rasmus Bonnevie on issue #328, credits to him.
import tensorflow as tf
from gpflow.kernels import Kernel
from gpflow.decors import params_as_tensors

class MultiKernel(Kernel):
    """Abstract class for MultiOutput Kernels.

    This abstract kernel assumes input X where the first column is a
    series of integer indices and the remaining dimensions are
    unconstrained. Multikernels are designed to handle outputs from
    different Gaussian processes, specifically in the case where they
    are not independent and where they can be observed independently.
    This abstract class implements the functionality necessary to split
    the observations into different cases and reorder the final kernel
    matrix appropriately.
    """
    def __init__(self, input_dim, output_dim, active_dims=None, name=None):
        Kernel.__init__(self, input_dim, active_dims, name)
        self.output_dim = output_dim

    def subK(self, indexes, X, X2=None):
        return NotImplementedError

    def subKdiag(self, indexes, X):
        return NotImplementedError

    @params_as_tensors
    def K(self, X, X2=None):
        # X, X2 = self._slice(X, X2)
        Xindex = tf.cast(X[:, 0], tf.int32)  # find group indices

        Xparts, Xsplitn, Xreturn = self._splitback(X[:, 1:], Xindex)

        if X2 is None:
            X2, X2parts, X2return, X2splitn = (X, Xparts, Xreturn, Xsplitn)
        else:
            X2index = tf.cast(X2[:, 0], tf.int32)
            X2parts, X2splitn, X2return = self._splitback(X2[:, 1:], X2index)

        #original
        #Find out what happens when there's an empty index for output_dim
        # construct kernel matrix for index-sorted data (stacked Xparts)
        blocks = []
        for i in range(self.output_dim):
            row_i = []
            for j in range(self.output_dim):
                row_i.append(self.subK((i, j), Xparts[i], X2parts[j]))
            blocks.append(tf.concat(row_i, 1))
        Ksort = tf.concat(blocks, 0)

        #new

        # Ksort = self.subK((tf.unstack(Xsplitn),tf.unstack(X2splitn)), X[:,1:], X2[:,1:])

        #ORIGINAL
        # split matrix into chunks, then stitch them together in correct order
        Ktmp = self._reconstruct(Ksort, Xsplitn, Xreturn)
        KT = self._reconstruct(tf.transpose(Ktmp), X2splitn, X2return)
        Kout = tf.transpose(KT, name='K')
        return Kout
        #ORIGINAL

        # return Ksort

    def Kdiag(self, X):
        # X, _ = self._slice(X, None)
        Xindex = tf.cast(X[:, 0], tf.int32)  # find recursion level indices
        Xparts, Xsplitn, Freturn = self._splitback(X[:, 1:], Xindex)

        subdiags = []
        for index, data in enumerate(Xparts):
            subdiags.append(self.subKdiag(index, Xparts[index]))
        Kd = tf.concat(subdiags, 0)
        return self._reconstruct(Kd, Xsplitn, Freturn)

    def _splitback(self, data, indices):
        """Apply dynamic_partitioning and calculate necessary statistics
        for inverse mapping."""
        # split data based on indices
        parts = tf.dynamic_partition(data, indices, self.output_dim)

        # to be able to invert the splitting, we need:
        # the size of each data split
        splitnum = tf.stack([tf.shape(x)[0] for x in parts])
        # indices to invert dynamic_part
        goback = tf.dynamic_partition(tf.range(tf.shape(data)[0]), indices, self.output_dim)
        return (parts, splitnum, goback)

    def _reconstruct(self, K, splitnum, goback):
        """Use quantities from splitback to invert a dynamic_partition."""
        tmp = tf.split(K, splitnum, axis=0)
        return tf.dynamic_stitch(goback, tmp)  # stitch
