import numpy as np
from .data import Data

class DataSet:
    def __init__(self, *args):
        """
        DataSet is a class that holds multiple Data objects as channels.

        Args:
            *args (Data,DataSet,list,dict): Accepts multiple arguments, each of which should be either a Data object, a list of Data objects or a dictionary of Data objects. Each Data object will be added to the list of channels. In case of a dictionary, the key will set the name of the Data object. If a DataSet is passed, its channels will be added.
        """

        self.channels = []
        for arg in args:
            if isinstance(arg, Data):
                self.channels.append(arg)
            elif isinstance(arg, DataSet):
                for val in arg.channels:
                    self.channels.append(val)
            elif isinstance(arg, list) and all(isinstance(val, Data) for val in arg):
                for val in arg:
                    self.channels.append(val)
            elif isinstance(arg, dict) and all(isinstance(val, Data) for val in arg.values()):
                for key, val in arg.items():
                    val.name = key
                    self.channels.append(val)

    def __iter__(self):
        return self.channels.__iter__()

    def __len__(self):
        return len(self.channels)

    def __getitem__(self, key):
        return self.channels[key]

    def get_input_dims(self):
        """
        Return the input dimensions per channel.

        Returns:
            list: List of input dimensions per channel.
        """
        return [channel.get_input_dims() for channel in self.channels]

    def get_output_dims(self):
        """
        Return the output dimensions of the dataset, i.e. the number of channels.

        Returns:
            int: Output dimensions.
        """
        return len(self.channels)

    def get_names(self):
        """
        Return the names of the channels.

        Returns:
            list: List of names.
        """
        return [channel.name for channel in self.channels]

    def get(self, index):
        """
        Return Data objects for a channel.

        Args:
            index (int,string): Index or name of the channel.

        Returns:
            Data: Channel data.
        """
        if isinstance(index, int):
            if index < len(self.channels):
                return self.channels[index]
        elif isinstance(index, str):
            for channel in self.channels:
                if channel.name == index:
                    return channel
        raise ValueError("channel '%d' does not exist in DataSet" % (index))
    
    def get_data(self):
        """
        Returns the observations.

        Returns:
            ndarray: X data of shape (output_dims,n,input_dims).
            ndarray: Y data of shape (output_dims,n).
        """
        return np.array([channel.get_data()[0] for channel in self.channels]), np.array([channel.get_data()[1] for channel in self.channels])
    
    def get_all(self):
        """
        Returns all observations (including removed observations).

        Returns:
            ndarray: X data of shape (output_dims,n,input_dims).
            ndarray: Y data of shape (output_dims,n).
        """
        return np.array([channel.get_all()[0] for channel in self.channels]), np.array([channel.get_all()[1] for channel in self.channels])

    def get_deleted(self):
        """
        Returns the removed observations.

        Returns:
            ndarray: X data of shape (output_dims,n,input_dims).
            ndarray: Y data of shape (output_dims,n).
        """
        return np.array([channel.get_deleted()[0] for channel in self.channels]), np.array([channel.get_deleted()[1] for channel in self.channels])
    
    def get_pred(self, name, sigma=2):
        x = []
        mu = []
        lower = []
        upper = []
        for channel in self.channels:
            channel_x, channel_mu, channel_lower, channel_upper = channel.get_pred(name, sigma)
            x.append(channel_x)
            mu.append(channel_mu)
            lower.append(channel_lower)
            upper.append(channel_upper)
        return np.array(x), np.array(mu), np.array(lower), np.array(upper)

    def set_pred(self, xs):
        """
        Set the prediction range directly.

        Args:
            xs (list,np.ndarray,dict): Array of shape (n) or (n,input_dims) with input values to predict at.

        Examples:
            >>> data.set_pred([5.0, 5.5, 6.0, 6.5, 7.0])
        """
        if isinstance(xs, list) or isinstance(xs, np.ndarray):
            if len(xs) != len(self.channels):
                raise ValueError("prediction x expected to be a list of shape (output_dims,)")

            for i, channel in enumerate(self.channels):
                channel.set_pred(xs[i])
        elif isinstance(xs, dict):
            for name in xs:
                self.get(name).set_pred(xs[name])
        else:
            raise ValueError("prediction x expected to be a list, ndarray or dict")
    
    def get_nyquist_estimation(self):
        return [channel.get_nyquist_estimation() for channel in self.channels]
    
    def get_bnse_estimation(self, Q):
        amplitudes = []
        means = []
        variances = []
        for channel in self.channels:
            channel_amplitudes, channel_means, channel_variances = channel.get_bnse_estimation(Q)
            amplitudes.append(channel_amplitudes)
            means.append(channel_means)
            variances.append(channel_variances)
        return np.array(amplitudes), np.array(means), np.array(variances)

    def to_kernel(self):
        """
        Return the data vectors in the format as used by the kernels.

        Returns:
            ndarray: X data of shape (n,. TODO
            ndarray: Y data.
        """
        x = [channel.X[channel.mask] for channel in self.channels]
        y = [channel.Y[channel.mask] for channel in self.channels]

        chan = [i * np.ones(len(x[i])) for i in range(len(x))]
        chan = np.concatenate(chan).reshape(-1, 1)
        
        x = np.concatenate(x)
        x = np.concatenate((chan, x), axis=1)
        if y == None:
            return x

        y = np.concatenate(y).reshape(-1, 1)
        return x, y

    def to_kernel_pred(self):
        """
        Return the prediction vectors in the format as used by the kernels.

        Returns:
            ndarray: X data of shape (n,. TODO
        """
        x = [channel.X_pred for channel in self.channels]

        chan = [i * np.ones(len(x[i])) for i in range(len(x))]
        chan = np.concatenate(chan).reshape(-1, 1)

        x = np.concatenate(x)
        x = np.concatenate((chan, x), axis=1)
        return x

    def from_kernel_pred(self, name, mu, var):
        N = [len(channel.X_pred) for channel in self.channels]
        if len(mu) != len(var) or sum(N) != len(mu):
            raise ValueError("prediction mu or var different length from prediction x")

        i = 0
        for idx in range(len(self.channels)):
            self.channels[idx].Y_mu_pred[name] = mu[i:i+N[idx]].reshape(1, -1)[0]
            self.channels[idx].Y_var_pred[name] = var[i:i+N[idx]].reshape(1, -1)[0]
            i += N[idx]

    def copy(self):
        """
        Make a deep copy of DataSet.

        Returns:
            DataSet: DataSet.
        """
        return copy.deepcopy(self)
