import numpy as np

class Data:
    def __init__(self):
        self.X = []
        self.Y = []
        self.X_all = []
        self.Y_all = []
        self.F = {}
        self.dims = 1
        self.channel_names = []

    def add(self, X, Y, name=None):
        if type(X) is not np.ndarray or type(Y) is not np.ndarray:
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

    def set_latent_function(self, channel, f):
        channel = self.get_channel_index(channel)
        self.F[channel] = f

    def get_input_dimensions(self):
        return self.dims

    def get_output_dimensions(self):
        return len(self.X)

    def get_channel_index(self, channel):
        if isinstance(channel, str):
            if channel not in self.channel_names:
                raise Exception("channel '%s' does not exist" % (channel))
            channel = self.channel_names.index(channel)
        if len(self.X) <= channel:
            raise Exception("channel %d does not exist" % (channel))
        return channel
    
    def get_channel_size(self, channel):
        channel = self.get_channel_index(channel)
        return self.X[channel].shape[0]

    def get_channel_sizes(self):
        sizes = []
        for x in self.X:
            sizes.append(x.shape[0])
        return sizes
    
    def get_observations(self):
        chan = []
        for channel in range(len(self.X)):
            chan.append(channel * np.ones(len(self.X[channel])))
        chan = np.concatenate(chan)
        x = np.concatenate(self.X)
        y = np.concatenate(self.Y)
        return np.stack((chan, x), axis=1), y.reshape(-1, 1)
    
    def remove_range(self, channel, start=None, end=None):
        channel = self.get_channel_index(channel)

        if start == None:
            start = self.X[channel][0]
        if end == None:
            end = self.X[channel][-1]

        idx = np.where(np.logical_and(self.X[channel] >= start, self.X[channel] <= end))
        self.X[channel] = np.delete(self.X[channel], idx)
        self.Y[channel] = np.delete(self.Y[channel], idx)
    
    def remove_relative_range(self, channel, start, end):
        channel = self.get_channel_index(channel)

        xmin = self.X_all[channel][0]
        xmax = self.X_all[channel][-1]
        start = xmin + np.round(max(0.0, start) * (xmax-xmin))
        end = xmin + np.round(min(1.0, end) * (xmax-xmin))

        self.remove_range(channel, start, end)

    def remove_random_ranges(self, channel, ranges, range_size):
        channel = self.get_channel_index(channel)
        if ranges < 1 or range_size < 1:
            return

        n = self.X[channel].shape[0] - ranges*range_size
        if n <= 0:
            raise Exception("no data left after removing ranges")

        locs = np.round(np.sort(np.random.rand(ranges)) * n)
        for i in range(len(locs)):
            loc = int(locs[i] + i * range_size)
            self.X[channel] = np.delete(self.X[channel], np.arange(loc, loc+range_size))
            self.Y[channel] = np.delete(self.Y[channel], np.arange(loc, loc+range_size))

