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

    ################################################################

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
    
    def get_ts_observations(self):
        chan = []
        for channel in range(len(self.X)):
            chan.append(channel * np.ones(len(self.X[channel])))
        chan = np.concatenate(chan)
        x = np.concatenate(self.X)
        y = np.concatenate(self.Y)
        return np.stack((chan, x), axis=1), y.reshape(-1, 1)
    
    def get_observations(self, channel):
        channel = self.get_channel_index(channel)
        return self.X[channel], self.Y[channel]
    
    def get_all_observations(self, channel):
        channel = self.get_channel_index(channel)
        return self.X_all[channel], self.Y_all[channel]

    def get_deleted_observations(self, channel):
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
    
    # remove observations randomly on the whole range for a certain channel, either n observations are removed, or a percentage
    def remove_randomly(self, channel, n=None, pct=None):
        channel = self.get_channel_index(channel)

        if n == None:
            if pct == None:
                n = 0
            else:
                n = int(pct * len(self.X[channel]))

        idx = np.random.randint(0, len(self.X[channel]), n)
        self.X[channel] = np.delete(self.X[channel], idx)
        self.Y[channel] = np.delete(self.Y[channel], idx)
    
    # remove observations on a channel in the interval [start,end]
    def remove_range(self, channel, start=None, end=None):
        channel = self.get_channel_index(channel)

        if start == None:
            start = self.X[channel][0]
        if end == None:
            end = self.X[channel][-1]

        idx = np.where(np.logical_and(self.X[channel] >= start, self.X[channel] <= end))
        self.X[channel] = np.delete(self.X[channel], idx)
        self.Y[channel] = np.delete(self.Y[channel], idx)
    
    # remove observations on a channel between start and end as a percentage of the number of observations.
    # start and end are in the range [0,1], where 0 is the first observation, 1 the last, and 0.5 the middle observation 
    def remove_relative_range(self, channel, start, end):
        channel = self.get_channel_index(channel)

        xmin = self.X_all[channel][0]
        xmax = self.X_all[channel][-1]
        start = xmin + np.round(max(0.0, start) * (xmax-xmin))
        end = xmin + np.round(min(1.0, end) * (xmax-xmin))

        self.remove_range(channel, start, end)

    # remove a number of ranges on a channel, where ranges is the number of ranges to remove, and range_size the width
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

    ################################################################

    # TODO: spectral info

