import csv
import copy
import numpy as np
from mogptk.bnse import *
from scipy.signal import lombscargle, find_peaks

# TODO: as X data may not be sorted (think 2D case), see if remove* and predict* functions work
class Data:
    """
    Class that holds all the observations and latent functions.

    It has functionality to add or remove observations in several ways,
    for example removing data ranges from the observations to simulate sensor failure.
    """
    def __init__(self):
        self.X = [] # for each channel the shape is (n, input_dims) with n the number of data points
        self.Y = [] # for each channel the shape is (n) with n the number of data points
        self.X_all = []
        self.Y_all = []
        self.F = {}
        self.dims = 1
        self.channel_names = []

    def __str__(self):
        return "Input dims: %d\nOutput dims: %d\nX: %s\nY: %s" % (self.get_input_dims(), self.get_output_dims(), self.X, self.Y)

    def _encode(self):
        return {
            'X': np.array(self.X),
            'Y': np.array(self.Y),
            'X_all': np.array(self.X_all),
            'Y_all': np.array(self.Y_all),
            #'F': self.F, # TODO
            'dims': self.dims,
            'channel_names': self.channel_names,
        }

    def _decode(d):
        self = Data()
        self.X = list(d['X'])
        self.Y = list(d['Y'])
        self.X_all = list(d['X_all'])
        self.Y_all = list(d['Y_all'])
        #self.F = d['F']
        self.dims = d['dims']
        self.channel_names = d['channel_names']
        return self

    def load_csv(self, filename, x_cols, y_cols, **kwargs):
        input_dims = 1
        if isinstance(x_cols, list) and all(isinstance(item, str) for item in x_cols):
            input_dims = len(x_cols)
        elif isinstance(x_cols, str):
            x_cols = [x_cols]
        else:
            raise Exception("x_cols must be string or list of strings")
        
        output_dims = 1
        if isinstance(y_cols, list) and all(isinstance(item, str) for item in y_cols):
            output_dims = len(y_cols)
        elif isinstance(y_cols, str):
            y_cols = [y_cols]
        else:
            raise Exception("y_cols must be string or list of strings")

        with open(filename, mode='r') as csv_file:
            rows = list(csv.DictReader(csv_file, **kwargs))
            
            X = np.empty((len(rows), input_dims))
            Y = np.empty((len(rows), output_dims))
            for j, row in enumerate(rows):
                for i, x_col in enumerate(x_cols):
                    X[j,i] = row[x_col]
                for i, y_col in enumerate(y_cols):
                    Y[j,i] = row[y_col]

            for i, y_col in enumerate(y_cols):
                self.add(X, Y[:,i], name=y_col)
    
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

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim == 2:
            if len(self.X) == 0:
                self.dims = X.shape[1]
            elif self.dims != X.shape[1]:
                raise Exception("X must have the same input dimensions for all channels")
        else:
            raise Exception("X must be either a one or two dimensional array of data")

        if Y.ndim != 1:
            raise Exception("Y must be a one dimensional array of data")
        if X.shape[0] != Y.shape[0]:
            raise Exception("X and Y must be of the same length")
        
        if name == None:
            name = str(len(self.channel_names))
        
        for channel_name in self.channel_names:
            if name == channel_name:
                raise Exception("channel name '%s' already exists" % (name))

        self.X.append(X)
        self.Y.append(Y)
        self.X_all.append(X)
        self.Y_all.append(Y)
        self.channel_names.append(name)

    def set_function(self, channel, f):
        """
        Sets a (latent) function corresponding to the channel.

        This is used for plotting functionality and is optional.
        """
        channel = self.get_channel_index(channel)
        self.F[channel] = f

    def add_function(self, f, n, start, end, var=0.0, name=None):
        """
        Adds a new channel.

        It is done by picking n observations on a (latent) function f, in the
        interval [start,end]. Optionally, it adds Gaussian noise of variance var
        to Y (the dependant variable) and allows for naming the channel (see add()).
        """
        x = np.sort(np.random.uniform(start, end, n))
        y = f(x) + np.random.normal(0.0, var, n)

        self.F[len(self.X)] = f
        self.add(x, y, name)

    def copy(self):
        return copy.deepcopy(self)

    def normalize(self):
        """
        Normalize output data for all channels
        """

        ymin = np.inf
        ymax = -np.inf
        for channel in range(self.get_output_dims()):
            ymin = np.amin([ymin, np.amin(self.Y[channel])])
            ymax = np.amax([ymax, np.amax(self.Y[channel])])
        
        for channel in range(self.get_output_dims()):
            self.Y[channel] = (self.Y[channel]-ymin) / (ymax-ymin)
            self.Y_all[channel] = (self.Y_all[channel]-ymin) / (ymax-ymin)

    ################################################################

    def get_input_dims(self):
        """
        Returns the input dimensions, where 
        length of the second dimension for X and Y when using add().
        """
        return self.dims

    def get_output_dims(self):
        """
        Returns the output dimensions (number of channels) of the data.
        """
        return len(self.X)

    def get_channel_index(self, channel):
        """
        Returns the channel index for a given channel name and checks if it exists.

        Args:
            channel (str, int): Channel to set prediction, can be either a string with the name
                of the channel or a integer with the index.
        
        Returns:
            Integer with number of channels
        """
        if isinstance(channel, str):
            if channel not in self.channel_names:
                raise Exception("channel '%s' does not exist" % (channel))
            channel = self.channel_names.index(channel)
        if len(self.X) <= channel:
            raise Exception("channel %d does not exist" % (channel))
        return channel
    
    def get_channel_size(self, channel):
        """
        Returns the number of observations for a channel.
        """
        channel = self.get_channel_index(channel)
        return self.X[channel].shape[0]

    def get_channel_sizes(self):
        """
        Returns the number of observations for all channels as a list.
        """
        sizes = []
        for x in self.X:
            sizes.append(x.shape[0])
        return sizes
    
    def get_obs(self, channel):
        """
        Returns the observations for a given channel.
        """
        channel = self.get_channel_index(channel)
        return self.X[channel], self.Y[channel]
    
    def get_all_obs(self, channel):
        """
        Returns all observations (including removed observations) for a given channel.
        """
        channel = self.get_channel_index(channel)
        return self.X_all[channel], self.Y_all[channel]

    def get_del_obs(self, channel):
        """
        Returns the removed observations for a given channel.
        """
        channel = self.get_channel_index(channel)

        js = []
        for i in range(len(self.X[channel])):
            x = self.X[channel][i]
            y = self.Y[channel][i]
            j = np.where(self.X_all[channel] == x)[0]
            if len(j) == 1 and self.Y_all[channel][j[0]] == y:
                js.append(j[0])

        X_removed = np.delete(self.X_all[channel], js, axis=0)
        Y_removed = np.delete(self.Y_all[channel], js, axis=0)
        return X_removed, Y_removed

    ################################################################
    
    def remove_randomly(self, channel, n=None, pct=None):
        """
        Removes observations randomly on the whole range for a certain channel.

        Either a number observations are removed, or a percentage of the observations.

        Args:
            channel (str, int): Channel to set prediction, can be either a string with the name
                of the channel or a integer with the index.

            n (int, optional): Number of observations to randomly remove.

            pct (float[0, 1], optional): Percentage of observations to remove.

            If neither 'n' or 'pct' are passed, 'n' is set to 0.
        """
        channel = self.get_channel_index(channel)

        if n == None:
            if pct == None:
                n = 0
            else:
                n = int(pct * self.X[channel].shape[0])

        idx = np.random.choice(self.X[channel].shape[0], n, replace=False)
        self.X[channel] = np.delete(self.X[channel], idx, 0)
        self.Y[channel] = np.delete(self.Y[channel], idx, 0)
    
    def remove_range(self, channel, start=None, end=None):
        """
        Removes observations on a channel in the interval [start,end].
        
        Args:
            channel (str, int): Channel to set prediction, can be either a string with the name
                of the channel or a integer with the index.

            start (float, optional): Value in input space to erase from. Default to first
                value in training points.

            end (float, optional): Value in input space to erase to. Default to last 
                value in training points.

        """
        channel = self.get_channel_index(channel)

        if start == None:
            start = self.X[channel][0]
        if end == None:
            end = self.X[channel][-1]

        idx = np.where(np.logical_and(self.X[channel] >= start, self.X[channel] <= end)) # TODO: X may not be sorted or multi dimensional
        self.X[channel] = np.delete(self.X[channel], idx, 0)
        self.Y[channel] = np.delete(self.Y[channel], idx, 0)
    
    def remove_relative_range(self, channel, start, end):
        """
        Removes observations on a channel between start and end as a percentage of the
        number of observations.

        Args:
            channel (str, int): Channel to set prediction, can be either a string with the name
                of the channel or a integer with the index.

            start (float in [0, 1]): Start of prediction to remove.

            end (flaot in [0, 1]): End of prediction to remove.

        Start and end are in the range [0,1], where 0 is the first observation,
        1 the last, and 0.5 the middle observation.
        """
        channel = self.get_channel_index(channel)

        xmin = self.X_all[channel][0]
        xmax = self.X_all[channel][-1]
        start = xmin + np.round(max(0.0, start) * (xmax-xmin))
        end = xmin + np.round(min(1.0, end) * (xmax-xmin))

        self.remove_range(channel, start, end)

    def remove_random_ranges(self, channel, n, size):
        """
        Removes a number of ranges on a channel.

        Args:
            channel (str, int): Channel to set prediction, can be either a string with the name
                of the channel or a integer with the index.

            n (int): Number of ranges to remove.

            size (int): Width of ranges to remove.
        """
        channel = self.get_channel_index(channel)
        if n < 1 or size < 1:
            return

        m = self.X[channel].shape[0] - n*size
        if m <= 0:
            raise Exception("no data left after removing ranges")

        locs = np.round(np.sort(np.random.rand(n)) * m)
        for i in range(len(locs)):
            loc = int(locs[i] + i * size)
            self.X[channel] = np.delete(self.X[channel], np.arange(loc, loc+size), 0)
            self.Y[channel] = np.delete(self.Y[channel], np.arange(loc, loc+size), 0)

    ################################################################

    def get_nyquist_estimation(self):
        """
        Estimate nyquist frequency for each channel  by taking
        0.5/min dist of points.

        Returns:
            Numpy array of length equal to number of channels.
        """
        input_dims = self.get_input_dims()
        output_dims = self.get_output_dims()

        nyquist = np.empty((output_dims, input_dims))
        for channel in range(self.get_output_dims()):
            for i in range(self.get_input_dims()):
                x = self.X[channel][i]
                dist = np.min(np.abs(x[1:]-x[:-1]))
                nyquist[channel,i] = 0.5/dist
        return nyquist

    def get_bnse_estimation(self, Q=1):
        """
        Peaks estimation using BNSE (Bayesian nonparametric espectral estimation)

        ** Only valid to single input dimension**
        """
        input_dims = self.get_input_dims()
        output_dims = self.get_output_dims()

        freqs = np.zeros((output_dims, input_dims, Q))
        amps = np.zeros((output_dims, input_dims, Q))

        nyquist = self.get_nyquist_estimation()
        for channel in range(self.get_output_dims()):
            for i in range(self.get_input_dims()):
                bnse = bse(self.X[channel][:,i], self.Y[channel])
                bnse.set_freqspace(nyquist[channel,i], dimension=1000)
                bnse.train()
                bnse.compute_moments()

                peaks, amplitudes = bnse.get_freq_peaks()
                if len(peaks) == 0:
                    continue

                peaks = np.array([peak for _, peak in sorted(zip(amplitudes, peaks), key=lambda pair: pair[0], reverse=True)])
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
                
                freqs[channel,i] = 2*np.pi*peaks
                amps[channel,i] = amplitudes
        return freqs, amps

    def get_ls_estimation(self, Q=1, n_ls=10000):
        """
        Peak estimation using Lomb Scargle.
        ***Only for 1 channel for the moment***
        To-Do: support for multiple channels.

        Args:
            Q (int): Number of components.
            n_ls (int): Number of points for Lomb Scargle,
                default to 10000.

        ** Only valid to single input dimension **
        """
        freqs = []
        amps = []

        # angular freq
        nyquist = np.array(self.get_nyquist_estimation()) * 2 * np.pi

        for channel in range(self.get_output_dims()):
            freq_space = np.linspace(1e-6, nyquist[channel], n_ls)
            pgram = lombscargle(self.X[channel], self.Y[channel], freq_space)
            peaks_index, _ = find_peaks(pgram)

            freqs_peaks = freq_space[peaks_index]
            amplitudes = pgram[peaks_index]

            peaks = np.array([(amp, peak) for amp, peak in sorted(zip(amplitudes, freqs_peaks), key=lambda pair: pair[0], reverse=True)])

            if Q < len(peaks):
                peaks = peaks[:Q]
            # if there is less peaks than components
            elif len(peaks) != 0:
                i = 0
                n = len(peaks)
                while Q > len(peaks):
                    peaks = np.r_[peaks, peaks[i] + np.random.standard_normal(2)]
                    i = (i+1) % n

            # TODO: use input dims
            freqs.append(np.expand_dims(peaks[:, 1], 0))
            amps.append(np.expand_dims(peaks[:, 0], 0))

        return freqs, amps
