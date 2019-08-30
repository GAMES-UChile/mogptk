import csv
import copy
import inspect
import dill
import numpy as np
from mogptk.bnse import *
from scipy.signal import lombscargle, find_peaks
import dateutil, datetime
import matplotlib
import matplotlib.pyplot as plt
import re
    
duration_regex = re.compile(
    r'^((?P<years>[\.\d]+?)y)?'
    r'^((?P<months>[\.\d]+?)M)?'
    r'^((?P<weeks>[\.\d]+?)w)?'
    r'^((?P<days>[\.\d]+?)d)?'
    r'((?P<hours>[\.\d]+?)h)?'
    r'((?P<minutes>[\.\d]+?)m)?'
    r'((?P<seconds>[\.\d]+?)s)?$')

def parse_duration(s):
    x = duration_regex.match(s)
    if x == None:
        raise Exception('duration string must be of the form 2h45m, allowed characters: (y)ear, (M)onth, (w)eek, (d)ay, (h)our, (m)inute, (s)econd')

    print(x)

class FormatNumber:
    def format(val):
        return val

    def parse(val, loc=None):
        try:
            return float(val)
        except ValueError:
            if loc == None:
                raise Exception("could not convert input to number")
            else:
                raise Exception("could not convert input to number at %s" % (loc))

    def parse_duration(val, loc=None):
        return FormatNumber.parse(val, loc)

    def scale(maxfreq=None):
        return 1, None

class FormatDate:
    def format(val):
        return datetime.datetime.utcfromtimestamp(val*3600*24).strftime('%Y-%m-%d')

    def parse(val, loc=None):
        try:
            return (dateutil.parser.parse(val) - datetime.datetime(1970,1,1)).total_seconds()/3600/24
        except ValueError:
            if loc == None:
                raise Exception("could not convert input to date")
            else:
                raise Exception("could not convert input to date at %s" % (loc))

    def parse_duration(val):
        if isinstance(val, int):
            return val
        if isinstance(val, str):
            return parse_duration(val)
        raise Exception("could not convert input to duration")

    def scale(maxfreq=None):
        if maxfreq == 'year':
            return 356.2425, 'year'
        if maxfreq == 'month':
            return 30.4369, 'month'
        if maxfreq == None or maxfreq == 'day':
            return 1, 'day'
        if maxfreq == 'hour':
            return 1/24, 'hour'
        if maxfreq == 'minute':
            return 1/24/60, 'minute'
        if maxfreq == 'second':
            return 1/24/3600, 'second'

class FormatDateTime:
    def format(val):
        return datetime.datetime.utcfromtimestamp(val).strftime('%Y-%m-%d %H:%M')

    def parse(val, loc=None):
        try:
            return (dateutil.parser.parse(val) - datetime.datetime(1970,1,1)).total_seconds()
        except ValueError:
            if loc == None:
                raise Exception("could not convert input to datetime")
            else:
                raise Exception("could not convert input to datetime at %s" % (loc))

    def parse_duration(val):
        if isinstance(val, int):
            return val
        if isinstance(val, str):
            return parse_duration(val)
        raise Exception("could not convert input to duration")

    def scale(maxfreq=None):
        if maxfreq == 'year':
            return 3600*24*356.2425, 'year'
        if maxfreq == 'month':
            return 3600*24*30.4369, 'month'
        if maxfreq == 'day':
            return 3600*24, 'day'
        if maxfreq == 'hour':
            return 3600, 'hour'
        if maxfreq == 'minute':
            return 60, 'minute'
        if maxfreq == None or maxfreq == 'second':
            return 1, 'second'

class TransformDetrend:
    def __init__(self, data, channel):
        if data.get_input_dims() != 1:
            raise Exception("can only remove ranges on one dimensional input data")

        # TODO: do not assume order in X
        self.coef = np.polyfit(data.X[channel][:,0], data.Y[channel], 1)

    def forward(self, x, y):
        return y-self.coef[1]-self.coef[0]*x[:,0]
    
    def backward(self, x, y):
        return y+self.coef[1]+self.coef[0]*x[:,0]

class TransformNormalize:
    def __init__(self, data, channel):
        self.ymin = np.inf
        self.ymax = -np.inf
        for channel in range(self.get_output_dims()):
            self.ymin = np.amin([self.ymin, np.amin(self.Y[channel])])
            self.ymax = np.amax([self.ymax, np.amax(self.Y[channel])])

    def forward(self, x, y):
        return (y-self.ymin)/(self.ymax-self.ymin)
    
    def backward(self, x, y):
        return y*(self.ymax-self.ymin)+self.ymin

class TransformLog:
    def forward(x, y):
        return np.log(y)
    
    def backward(x, y):
        return np.exp(y)

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
        self.dims = None
        self.channel_names = []
        self.formatters = []
        self.input_labels = []
        self.output_labels = []
        self.transformations = []

    def __str__(self):
        return "Input dims: %d\nOutput dims: %d\nX: %s\nY: %s" % (self.get_input_dims(), self.get_output_dims(), self.X, self.Y)

    # TODO: update encode/decode with new data members
    def _encode(self):
        F = []
        keys = list(self.F.keys())
        for i in range(len(keys)):
            s = self.F[keys[i]]
            s = dill.dumps(s)
            s = str(s)
            F.append([keys[i], s])

        return {
            'X': np.array(self.X),
            'Y': np.array(self.Y),
            'X_all': np.array(self.X_all),
            'Y_all': np.array(self.Y_all),
            'F': F,
            'dims': self.dims,
            'channel_names': self.channel_names,
        }

    def _decode(d):
        self = Data()
        self.X = list(d['X'])
        self.Y = list(d['Y'])
        self.X_all = list(d['X_all'])
        self.Y_all = list(d['Y_all'])

        F = d['F']
        for f in F:
            key = f[0]
            s = f[1]
            s = eval(s)
            s = dill.loads(s)
            self.F[key] = s

        self.dims = d['dims']
        self.channel_names = d['channel_names']
        return self

    def _normalize_input_dims(self, x):
        if x == None:
            return x
        if isinstance(x, float):
            x = [x]
        elif isinstance(x, int):
            x = [float(x)]
        elif isinstance(x, str):
            x = [x]
        elif isinstance(x, np.ndarray):
            x = list(x)
        elif not isinstance(x, list):
            raise Exception("input should be a floating point, list or ndarray")
        if self.get_input_dims() != None and len(x) != self.get_input_dims():
            raise Exception("input must be a scalar for single-dimension input or a list of values for each input dimension")
        return x

    def set_labels(self, channel, input, output):
        if isinstance(input, str):
            input = [input]
        elif not isinstance(input, list) or not all(isinstance(item, str) for item in input):
            raise Exception("input labels must be list of strings")
        if not isinstance(output, str):
            raise Exception("output label must be string")

        if self.dims == None:
            raise Exception("set data first before setting labels")
        elif len(input) != self.dims:
            raise Exception("input labels must have the same input dimensions as the data")

        channel = self.get_channel_index(channel)
        self.input_labels[channel] = input
        self.output_labels[channel] = output

    def load_csv(self, filename, x_cols, y_cols, format={}, filter=None, name=None, **kwargs):
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

            def _to_number(val, row, col):
                try:
                    if col in format:
                        return format[col].parse(val, loc="row %d column %s" % (row+1, col)), True
                    else:
                        return FormatNumber.parse(val, loc="row %d column %s" % (row+1, col)), True
                except:
                    return np.nan, False
            
            X = np.empty((len(rows), input_dims))
            Y = np.empty((len(rows), output_dims))
            remove = []
            for j, row in enumerate(rows):
                if filter != None and not filter(row):
                    remove.append(j)
                    continue

                for i, x_col in enumerate(x_cols):
                    X[j,i], ok = _to_number(row[x_col], j+1, x_col)
                    if not ok:
                        remove.append(j)

                for i, y_col in enumerate(y_cols):
                    Y[j,i], ok = _to_number(row[y_col], j+1, y_col)
                    if not ok:
                        remove.append(j)

            X = np.delete(X, remove, 0)
            Y = np.delete(Y, remove, 0)

            fmts = []
            for x_col in x_cols:
                if x_col in format:
                    fmts.append(format[x_col])
                else:
                    fmts.append(FormatNumber)

            for i, y_col in enumerate(y_cols):
                channel_name = y_col
                if name is not None:
                    channel_name = name
                self.add(X, Y[:,i], name=channel_name, formatters=fmts)
                self.set_labels(channel_name, x_cols, y_col)
    
    def add(self, X, Y, name=None, formatters=None):
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
            if self.dims == None:
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

        if formatters == None:
            formatters = [FormatNumber] * X.shape[1]
        if len(formatters) != X.shape[1]:
            raise Exception("formatters must be defined for all input dimensions")

        self.X.append(X)
        self.Y.append(Y)
        self.X_all.append(X.copy())
        self.Y_all.append(Y.copy())
        self.channel_names.append(name)
        self.formatters.append(formatters)
        self.input_labels.append([''] * X.shape[1])
        self.output_labels.append('')
        self.transformations.append([])

    def _check_function(self, f):
        if not inspect.isfunction(f):
            raise Exception("F must be a function taking X as a parameter")

        sig = inspect.signature(f)
        if not len(sig.parameters) == 1:
            raise Exception("F must be a function taking X as a parameter")

    def set_function(self, channel, f):
        """
        Sets a (latent) function corresponding to the channel. The function must take one parameter X (shape (n,input_dims) and output Y (shape (n)).

        This is used for plotting functionality and is optional.
        """
        channel = self.get_channel_index(channel)
        self._check_function(f)
        self.F[channel] = f

    def load_function(self, f, n, start, end, var=0.0, name=None):
        """
        Adds a new channel.

        It is done by picking n observations on a (latent) function f, in the
        interval [start,end]. Optionally, it adds Gaussian noise of variance var
        to Y (the dependant variable) and allows for naming the channel (see add()).
        """
        self._check_function(f)
        
        start = self._normalize_input_dims(start)
        end = self._normalize_input_dims(end)

        if self.dims != None:
            input_dims = self.get_input_dims()
        else:
            input_dims = len(start)

        x = np.empty((n, input_dims))
        for i in range(input_dims):
            x[:,i] = np.random.uniform(start[i], end[i], n)
        x = np.sort(x, axis=0)

        y = f(x)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y[:,0]
        y += np.random.normal(0.0, var, n)

        self.add(x, y, name)
        self.F[len(self.X)-1] = f

    def copy(self):
        return copy.deepcopy(self)

    def transform(self, channels, transformer):
        if channels == '*':
            channels = range(self.get_output_dims())
        elif not isinstance(channels, list):
            channels = [channels]
        for channel in channels:
            t = transformer
            if '__init__' in vars(transformer):
                t = transformer(self, channel)

            self.Y[channel] = t.forward(self.X[channel], self.Y[channel])
            self.Y_all[channel] = t.forward(self.X_all[channel], self.Y_all[channel])
            if channel in self.F:
                f = self.F[channel]
                self.F[channel] = lambda x: t.forward(x, f(x))
    
    def filter(self, channels, start, end):
        if self.get_input_dims() != 1:
            raise Exception("can only filter on one dimensional input data")

        start = self.formatters[channel][0].parse(start)
        end = self.formatters[channel][0].parse(end)
        
        if channels == '*':
            channels = range(self.get_output_dims())
        elif not isinstance(channels, list):
            channels = [channels]
        for channel in channels:
            pass # TODO


    def aggregate(self, channels, duration):
        if self.get_input_dims() != 1:
            raise Exception("can only aggregate on one dimensional input data")

        duration = self.formatters[channel][0].parse_duration(duration)
        
        if channels == '*':
            channels = range(self.get_output_dims())
        elif not isinstance(channels, list):
            channels = [channels]
        for channel in channels:
            pass # TODO

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
        if channel == -1:
            channel = len(self.X)-1
        if len(self.X) <= channel or channel < 0:
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
            channel (str, int): Channel to set prediction, can be either a string
                with the name of the channel or a integer with the index.

            n (int, optional): Number of observations to randomly keep.

            pct (float[0, 1], optional): Percentage of observations to remove.

            If neither 'n' or 'pct' are passed, 'n' is set to 0.
        """
        channel = self.get_channel_index(channel)

        if n == None:
            if pct == None:
                n = 0
            else:
                n = int((1-pct) * self.X[channel].shape[0])

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
        if self.get_input_dims() != 1:
            raise Exception("can only remove ranges on one dimensional input data")

        channel = self.get_channel_index(channel)

        if start == None:
            start = np.min(self.X[channel][:,0])
        else:
            start = self.formatters[channel][0].parse(start)
        if end == None:
            end = np.max(self.X[channel][:,0])
        else:
            end = self.formatters[channel][0].parse(end)

        idx = np.where(np.logical_and(self.X[channel][:,0] >= start, self.X[channel][:,0] <= end))
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

            end (float in [0, 1]): End of prediction to remove.

        Start and end are in the range [0,1], where 0 is the first observation,
        1 the last, and 0.5 the middle observation.
        """
        if self.get_input_dims() != 1:
            raise Exception("can only remove ranges on one dimensional input data")

        channel = self.get_channel_index(channel)
        start = self.formatters[channel][0].parse(start)
        end = self.formatters[channel][0].parse(end)

        x_min = np.min(self.X[channel][:,0])
        x_max = np.max(self.X[channel][:,0])
        start = x_min + np.round(max(0.0, min(1.0, start)) * (x_max-x_min))
        end = x_min + np.round(max(0.0, min(1.0, end)) * (x_max-x_min))

        idx = np.where(np.logical_and(self.X[channel][:,0] >= start, self.X[channel][:,0] <= end))
        self.X[channel] = np.delete(self.X[channel], idx, 0)
        self.Y[channel] = np.delete(self.Y[channel], idx, 0)

    def remove_random_ranges(self, channel, n, size):
        """
        Removes a number of ranges on a channel. Makes only sense if your input X is sorted.

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
        Estimate nyquist frequency for each channel by taking
        0.5/min dist of points.

        Returns:
            Numpy array of length equal to number of channels.
        """
        input_dims = self.get_input_dims()
        output_dims = self.get_output_dims()

        nyquist = np.empty((input_dims, output_dims))
        for channel in range(self.get_output_dims()):
            for i in range(self.get_input_dims()):
                x = np.sort(self.X[channel][:,i])
                dist = np.abs(x[1:]-x[:-1]) # TODO: assumes X is sorted, use average distance instead of minimal distance?
                dist = np.min(dist[np.nonzero(dist)])
                nyquist[i,channel] = 0.5/dist
        return nyquist

    def get_bnse_estimation(self, Q=1):
        """
        Peaks estimation using BNSE (Bayesian Non-parametric Spectral Estimation)

        Returns:
            freqs, amps: Each one is a input_dim x n_channels x Q array with 
                the frequency values and amplitudes of the peaks.

        """
        input_dims = self.get_input_dims()
        output_dims = self.get_output_dims()

        freqs = np.zeros((output_dims, input_dims, Q))
        amps = np.zeros((output_dims, input_dims, Q))

        nyquist = self.get_nyquist_estimation()
        for channel in range(output_dims):
            for i in range(input_dims):
                x = self.X[channel][:,i]
                y = self.Y[channel]
                bnse = bse(x, y)
                bnse.set_freqspace(nyquist[i,channel], dimension=5000)
                bnse.train()
                bnse.compute_moments()

                peaks, amplitudes = bnse.get_freq_peaks() # TODO: get peak widths
                if len(peaks) == 0:
                    continue

                peaks = np.array([peak for _, peak in sorted(zip(amplitudes, peaks), key=lambda pair: pair[0], reverse=True)])
                amplitudes.sort()

                if Q < len(peaks):
                    peaks = peaks[:Q]
                    amplitudes = amplitudes[:Q]
                elif len(peaks) != 0:
                    j = 0
                    n = len(peaks)
                    while Q > len(peaks):
                        peaks = np.append(peaks, peaks[j] + (np.random.standard_t(3, 1) * 0.01)[0])
                        amplitudes = np.append(amplitudes, amplitudes[j])
                        j = (j+1) % n
                
                freqs[channel,i,:] = 2*np.pi*peaks
                amps[channel,i,:] = amplitudes
        return freqs / np.pi / 2, amps

    def get_ls_estimation(self, Q=1, n_ls=50000):
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
        input_dims = self.get_input_dims()
        output_dims = self.get_output_dims()

        freqs = np.zeros((output_dims, input_dims, Q))
        amps = np.zeros((output_dims, input_dims, Q))

        nyquist = self.get_nyquist_estimation() * 2 * np.pi
        for channel in range(output_dims):
            for i in range(input_dims):
                freq_space = np.linspace(0, nyquist[i,channel], n_ls+1)[1:]
                pgram = lombscargle(self.X[channel][:,i], self.Y[channel], freq_space)
                peaks_index, _ = find_peaks(pgram)

                freqs_peaks = freq_space[peaks_index]
                amplitudes = pgram[peaks_index]

                peaks = np.array([(amp, peak) for amp, peak in sorted(zip(amplitudes, freqs_peaks), key=lambda pair: pair[0], reverse=True)])

                if Q < len(peaks):
                    peaks = peaks[:Q]
                # if there is less peaks than components
                elif len(peaks) != 0:
                    j = 0
                    n = len(peaks)
                    while Q > len(peaks):
                        peaks = np.r_[peaks, peaks[j] + np.random.standard_normal(2)]
                        j = (j+1) % n

                freqs[channel,i,:] = peaks[:,1]
                amps[channel,i,:] = peaks[:,0]

        return freqs / np.pi / 2, amps
    
    def get_gm_estimation(self):
        # TODO: use sklearn.mixture.GaussianMixture to retrieve fitted gaussian mixtures to spectral data
        pass

    def plot(self, show=True, filename=None, title=None):
        sns.set(font_scale=2)
        sns.axes_style("darkgrid")
        sns.set_style("whitegrid")

        fig, axes = plt.subplots(self.get_output_dims(), self.get_input_dims(), figsize=(20, self.get_output_dims()*5), sharey=False, constrained_layout=True, squeeze=False)
        if title != None:
            fig.suptitle(title, fontsize=36)

        plotting_F = False
        plotting_all_obs = False
        for i in range(self.get_input_dims()):
            for channel in range(self.get_output_dims()):
                if channel in self.F:
                    n = len(self.X[channel][:,i])*10
                    x_min = np.min(self.X[channel][:,i])
                    x_max = np.max(self.X[channel][:,i])

                    x = np.zeros((n, self.get_input_dims())) # assuming other input dimensions are zeros
                    x[:,i] = np.linspace(x_min, x_max, n)
                    y = self.F[channel](x)

                    axes[channel, i].plot(x[:,i], y, 'r--', lw=1)
                    plotting_F = True

                axes[channel, i].plot(self.X[channel][:,i], self.Y[channel], 'k-')
                axes[channel, i].set_xlabel(self.input_labels[channel][i])
                axes[channel, i].set_ylabel(self.output_labels[channel])
                axes[channel, i].set_title(self.channel_names[channel], fontsize=30)
            
                formatter = matplotlib.ticker.FuncFormatter(lambda x,pos: self.formatters[channel][i].format(x))
                axes[channel, i].xaxis.set_major_formatter(formatter)

        # build legend
        if plotting_F:
            legend = []
            legend.append(plt.Line2D([0], [0], ls='-', color='k', label='Data'))
            legend.append(plt.Line2D([0], [0], ls='--', color='r', label='Latent function'))
            plt.legend(handles=legend, loc='best')

        if filename != None:
            plt.savefig(filename+'.pdf', dpi=300)
        if show:
            plt.show()

    def plot_spectrum(self, method='lombscargle', angular=False, per=None, maxfreq=None, show=True, filename=None, title=None):
        sns.set(font_scale=2)
        sns.axes_style("darkgrid")
        sns.set_style("whitegrid")

        fig, axes = plt.subplots(self.get_output_dims(), self.get_input_dims(), figsize=(20, self.get_output_dims()*5), sharey=False, constrained_layout=True, squeeze=False)
        if title != None:
            fig.suptitle(title, fontsize=36)

        for i in range(self.get_input_dims()):
            for channel in range(self.get_output_dims()):
                X_space = self.X[channel][:,i].copy()

                formatter = self.formatters[channel][i]
                factor, name = formatter.scale(per)
                if name != None:
                    axes[channel,i].set_xlabel('Frequency (1/'+name+')')
                else:
                    axes[channel,i].set_xlabel('Frequency')

                if not angular:
                    X_space *= 2 * np.pi
                X_space /= factor

                freq = maxfreq
                if freq == None:
                    dist = np.abs(X_space[1:]-X_space[:-1])
                    freq = 1/np.average(dist)

                X = np.linspace(0.0, freq, 10001)[1:]
                if method == 'lombscargle':
                    Y = lombscargle(X_space, self.Y[channel], X)
                else:
                    raise Exception('Periodogram method "%s" does not exist' % (method))

                axes[channel,i].plot(X, Y, 'k-')
                axes[channel,i].set_title(self.channel_names[channel], fontsize=30)
                axes[channel,i].set_yticks([])
                axes[channel,i].set_ylim(0, None)

        if filename != None:
            plt.savefig(filename+'.pdf', dpi=300)
        if show:
            plt.show()
