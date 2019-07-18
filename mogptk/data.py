import numpy as np
from mogptk.bnse import *

class Data:
    """
    Class that holds all the observations and latent functions.

    It has functionality to add or remove observations in several ways,
    for example removing data ranges from the observations to simulate sensor failure.
    
    Atributes:
    """
    def __init__(self):
        self.X = []
        self.Y = []
        self.X_all = []
        self.Y_all = []
        self.F = {}
        self.dims = 1
        self.channel_names = []

    def add(self, X, Y, name=None):
        """
        Adds a new channel with data set by X (input) and Y (output).


        Optionally, you can set a name to identify the channel and use that in
        any other function that requires a channel identifier.
        X and Y need to be of equal length. If X and Y are two dimensional,
        the second dimension will determine the input dimensionality of the channel.

        Args:
            X (list, ndarray):
            Y (list. ndarray):
            name (str, optional):
        """
        if isinstance(X, list):
            X = np.array(X)
        if isinstance(Y, list):
            Y = np.array(Y)
        if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
            raise Exception("X and Y must be numpy arrays")

        if X.ndim == 2:
            if len(self.X) == 0:
                self.dims = X.shape[1]
            elif self.dims != X.shape[1]:
                raise Exception("X must have the same input dimensions for all channels")
        elif X.ndim != 1:
            raise Exception("X must be either a one or two dimensional array of data")
        if Y.ndim != 1:
            raise Exception("Y must be a one dimensional array of data")
        if X.shape[0] != Y.shape[0]:
            raise Exception("X and Y must be of the same length")
        if X.ndim == 1:
            # TODO: what about multi-dimension input data?
            for i in range(len(X)-1):
                if X[i] >= X[i+1]:
                    raise Exception("X must be ordered from low to high")

        if name == None:
            name = str(len(self.channel_names))

        self.X.append(X)
        self.Y.append(Y)
        self.X_all.append(X)
        self.Y_all.append(Y)
        self.channel_names.append(name)

    def set_function(self, channel, f):
        """set_function sets a (latent) function corresponding to the channel. This is used for plotting functionality and is optional."""
        channel = self.get_channel_index(channel)
        self.F[channel] = f

    def add_function(self, f, n, start, end, var=0.0, name=None):
        """add_function adds a new channel by picking n observations on a (latent) function f, in the interval [start,end]. Optionally, it adds Gaussian noise of variance var to Y (the dependant variable) and allows for naming the channel (see add())."""
        x = np.sort(np.random.uniform(start, end, n))
        y = f(x) + np.random.normal(0.0, var, n)

        self.F[len(self.X)] = f
        self.add(x, y, name)

    ################################################################

    def get_input_dims(self):
        """get_input_dims returns the input dimensions (length of the second dimension for X and Y when using add())."""
        return self.dims

    def get_output_dims(self):
        """get_output_dims returns the output dimensions (number of channels) of the data."""
        return len(self.X)

    def get_channel_index(self, channel):
        """get_channel_index returns the channel index for a given channel name and checks if it exists."""
        if isinstance(channel, str):
            if channel not in self.channel_names:
                raise Exception("channel '%s' does not exist" % (channel))
            channel = self.channel_names.index(channel)
        if len(self.X) <= channel:
            raise Exception("channel %d does not exist" % (channel))
        return channel
    
    def get_channel_size(self, channel):
        """get_channel_size returns the number of observations for a channel."""
        channel = self.get_channel_index(channel)
        return self.X[channel].shape[0]

    def get_channel_sizes(self):
        """get_channel_sizes returns the number of observations for all channels as a list."""
        sizes = []
        for x in self.X:
            sizes.append(x.shape[0])
        return sizes
    
    def get_ts_obs(self):
        """get_ts_obs returns the flattened array format of all observations for use by TensorFlow. In particular this will be an array of all observations of all channels concatenated. For X each entry is a two element array consisting of the channel ID and the X value."""
        chan = []
        for channel in range(len(self.X)):
            chan.append(channel * np.ones(len(self.X[channel])))
        chan = np.concatenate(chan)
        x = np.concatenate(self.X)
        y = np.concatenate(self.Y)
        return np.stack((chan, x), axis=1), y.reshape(-1, 1)
    
    def get_obs(self, channel):
        """get_obs returns the observations for a given channel."""
        channel = self.get_channel_index(channel)
        return self.X[channel], self.Y[channel]
    
    def get_all_obs(self, channel):
        """get_all_obs returns all observations (including removed observations) for a given channel."""
        channel = self.get_channel_index(channel)
        return self.X_all[channel], self.Y_all[channel]

    def get_del_obs(self, channel):
        """get_del_obs returns the removed observations for a given channel."""
        channel = self.get_channel_index(channel)

        js = []
        for i in range(len(self.X[channel])):
            x = self.X[channel][i]
            y = self.Y[channel][i]
            j = np.where(self.X_all[channel] == x)[0]
            if len(j) == 1 and self.Y_all[channel][j[0]] == y:
                js.append(j[0])

        X_removed = np.delete(self.X_all[channel], js)
        Y_removed = np.delete(self.Y_all[channel], js)
        return X_removed, Y_removed

    ################################################################
    
    def remove_randomly(self, channel, n=None, pct=None):
        """remove_randomly removes observations randomly on the whole range for a certain channel, either n observations are removed, or a percentage of the observations."""
        channel = self.get_channel_index(channel)

        if n == None:
            if pct == None:
                n = 0
            else:
                n = int(pct * len(self.X[channel]))

        idx = np.random.randint(0, len(self.X[channel]), n)
        self.X[channel] = np.delete(self.X[channel], idx)
        self.Y[channel] = np.delete(self.Y[channel], idx)
    
    def remove_range(self, channel, start=None, end=None):
        """remove_range removes observations on a channel in the interval [start,end]."""
        channel = self.get_channel_index(channel)

        if start == None:
            start = self.X[channel][0]
        if end == None:
            end = self.X[channel][-1]

        idx = np.where(np.logical_and(self.X[channel] >= start, self.X[channel] <= end))
        self.X[channel] = np.delete(self.X[channel], idx)
        self.Y[channel] = np.delete(self.Y[channel], idx)
    
    def remove_relative_range(self, channel, start, end):
        """remove_relative_range removes observations on a channel between start and end as a percentage of the number of observations.
        Start and end are in the range [0,1], where 0 is the first observation, 1 the last, and 0.5 the middle observation."""
        channel = self.get_channel_index(channel)

        xmin = self.X_all[channel][0]
        xmax = self.X_all[channel][-1]
        start = xmin + np.round(max(0.0, start) * (xmax-xmin))
        end = xmin + np.round(min(1.0, end) * (xmax-xmin))

        self.remove_range(channel, start, end)

    def remove_random_ranges(self, channel, n, size):
        """remove_random_ranges removes a number of ranges on a channel, where n is the number of ranges to remove, and size the width."""
        channel = self.get_channel_index(channel)
        if n < 1 or size < 1:
            return

        m = self.X[channel].shape[0] - n*size
        if m <= 0:
            raise Exception("no data left after removing ranges")

        locs = np.round(np.sort(np.random.rand(n)) * m)
        for i in range(len(locs)):
            loc = int(locs[i] + i * size)
            self.X[channel] = np.delete(self.X[channel], np.arange(loc, loc+size))
            self.Y[channel] = np.delete(self.Y[channel], np.arange(loc, loc+size))

    ################################################################

    def get_nyquist_estimation(self):
        nyquist = []
        for channel in range(self.get_output_dims()):
            x = self.X[channel]
            dist = np.min(np.abs(x[1:]-x[:-1]))
            nyquist.append(0.5/dist)
        return nyquist

    def get_bnse_estimation(self, Q=1):
        freqs = []
        amps = []
        nyquist = self.get_nyquist_estimation()
        for channel in range(self.get_output_dims()):
            bnse = bse(self.X[channel], self.Y[channel])
            bnse.set_freqspace(nyquist[channel])
            bnse.train()
            bnse.compute_moments()

            peaks, amplitudes = bnse.get_freq_peaks()
            peaks = np.array([peak for _, peak in sorted(zip(amplitudes, peaks), key=lambda pair: pair[0])])
            amplitudes.sort()

            if Q < len(peaks):
                peaks = peaks[:Q]
                amplitudes = amplitudes[:Q]
            elif len(peaks) != 0:
                i = 0
                n = len(peaks)
                while Q > len(peaks):
                    peaks = np.append(peaks, peaks[i] + (np.random.standard_t(3, 1) * 0.01)[0])
                    amplitudes = np.append(amplitudes, amplitudes[i])
                    i = (i+1) % n

            # TODO: use input dims
            peaks = np.expand_dims(peaks, axis=0)
            amplitudes = np.expand_dims(amplitudes, axis=0)

            freqs.append(2*np.pi*peaks)
            amps.append(amplitudes)
        return freqs, amps

