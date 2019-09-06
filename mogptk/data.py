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
import pandas as pd
    
duration_regex = re.compile(
    r'^((?P<years>[\.\d]+?)y)?'
    r'((?P<months>[\.\d]+?)M)?'
    r'((?P<weeks>[\.\d]+?)w)?'
    r'((?P<days>[\.\d]+?)d)?'
    r'((?P<hours>[\.\d]+?)h)?'
    r'((?P<minutes>[\.\d]+?)m)?'
    r'((?P<seconds>[\.\d]+?)s)?$')

def _parse_duration_to_sec(s):
    x = duration_regex.match(s)
    if x == None:
        raise Exception('duration string must be of the form 2h45m, allowed characters: (y)ear, (M)onth, (w)eek, (d)ay, (h)our, (m)inute, (s)econd')

    sec = 0
    matches = x.groups()[1::2]
    if matches[0]:
        sec += float(matches[0])*356.2425*24*3600
    if matches[1]:
        sec += float(matches[1])*30.4369*24*3600
    if matches[2]:
        sec += float(matches[2])*7*24*3600
    if matches[3]:
        sec += float(matches[3])*24*3600
    if matches[4]:
        sec += float(matches[4])*3600
    if matches[5]:
        sec += float(matches[5])*60
    if matches[6]:
        sec += float(matches[6])
    return sec

class FormatNumber:
    """
    FormatNumber is the default formatter and takes regular floating point values as input.
    """
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
    """
    FormatDate is a formatter that takes date values as input, such as '2019-03-01', and stores values internally as days since 1970-01-01.
    """
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
            return _parse_duration_to_sec(val)/24/3600
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
    """
    FormatDateTime is a formatter that takes date and time values as input, such as '2019-03-01 12:30', and stores values internally as seconds since 1970-01-01.
    """
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
            return _parse_duration_to_sec(val)
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
    """
    TransformDetrend is a transformer that detrends the data.
    """
    def __init__(self, data):
        if data.get_input_dims() != 1:
            raise Exception("can only remove ranges on one dimensional input data")

        self.coef = np.polyfit(data.X[:,0], data.Y, 1)

    def forward(self, x, y):
        return y-self.coef[1]-self.coef[0]*x[:,0]
    
    def backward(self, x, y):
        return y+self.coef[1]+self.coef[0]*x[:,0]

class TransformNormalize:
    """
    TransformNormalize is a transformer that normalizes the data, so that all Y data is between 0 and 1.
    """
    def __init__(self, data):
        self.ymin = np.amin([self.ymin, np.amin(self.Y)])
        self.ymax = np.amax([self.ymax, np.amax(self.Y)])

    def forward(self, x, y):
        return (y-self.ymin)/(self.ymax-self.ymin)
    
    def backward(self, x, y):
        return y*(self.ymax-self.ymin)+self.ymin

class TransformLog:
    """
    TransformLog is a transformer that takes the log of the data. Make sure there is no negative data.
    """
    def forward(x, y):
        return np.log(y)
    
    def backward(x, y):
        return np.exp(y)

def LoadFunction(f, start, end, n, var=0.0, name=None, random=False):
    """
    LoadFunction loads a dataset from a given function y = f(x) + N(0,var). It will pick n data points between start and end for x, for which f is being evaluated. By default the n points are spread equally over the interval, with random=True they will be picked randomly.

    The function should take one argument x with shape (n,input_dims) and return y with shape (n). If your data has only one input dimension, you can use x[:,0] to select only the first (and only) input dimension.

    Args:
        f (function): function taking x with shape (n,input_dims) and returning shape (n) as y
        n (int): number of data points to pick between start and end
        start (float, list): define start of interval
        end (float, list): define end of interval
        var (float, optional): variance added to the output
        name (str, optional): name of dataset
        random (boolean): select points randomly between start and end (defaults to False)
    """
    # TODO: make work for multiple input dimensions, take n as a list

    start = _normalize_input_dims(start, None)
    input_dims = len(start)
    if input_dims != 1:
        raise Exception("can only load function with one dimensional input data")
    
    end = _normalize_input_dims(end, input_dims)
    _check_function(f, input_dims)

    x = np.empty((n, input_dims))
    for i in range(input_dims):
        if start[i] >= end[i]:
            if input_dims == 1:
                raise Exception("start must be lower than end")
            else:
                raise Exception("start must be lower than end for input dimension %d" % (i))

        if random:
            x[:,i] = np.random.uniform(start[i], end[i], n)
        else:
            x[:,i] = np.linspace(start[i], end[i], n)

    y = f(x)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:,0]
    y += np.random.normal(0.0, var, n)

    data = Data(x, y, name=name)
    data.set_function(f)
    return data

def LoadCSV(filename, x_cols, y_col, name=None, format={}, filter=None, **kwargs):
    """
    LoadCSV loads a dataset from a given CSV file. It loads in x_cols as the names of the input dimension columns, and y_col the name of the output column. Setting a formatter for a column will enable parsing for example date fields such as '2019-03-01'. A filter can be set to filter out data from the CSV, such as ensuring that another column has a certain value.

    Args:
        filename (str): CSV filename
        x_cols (str, list): name or names of X column(s) in CSV
        y_col (str): name of Y column in CSV
        name (str, optional): name of dataset
        format (dict, optional): dictionary with x_cols values as keys containing FormatNumber (default), FormatDate, FormetDateTime, ...
        filter (function, optional): function that takes row as argument, and returns True to keep the record.
    """

    input_dims = 1
    if isinstance(x_cols, list) and all(isinstance(item, str) for item in x_cols):
        input_dims = len(x_cols)
    elif isinstance(x_cols, str):
        x_cols = [x_cols]
    else:
        raise Exception("x_cols must be string or list of strings")
    
    if not isinstance(y_col, str):
        raise Exception("y_col must be string")

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
        Y = np.empty((len(rows)))
        remove = []
        for j, row in enumerate(rows):
            if filter != None and not filter(row):
                remove.append(j)
                continue

            for i, x_col in enumerate(x_cols):
                X[j,i], ok = _to_number(row[x_col], j+1, x_col)
                if not ok:
                    remove.append(j)

            Y[j], ok = _to_number(row[y_col], j+1, y_col)
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

        data = Data(X, Y, name=name, format=fmts)
        data.set_labels(x_col, y_col)
        return data

class Data:
    """
    Class that holds all the observations and latent functions.

    It has functionality to add or remove observations in several ways,
    for example removing data ranges from the observations to simulate sensor failure.
    """
    def __init__(self, X, Y, name=None, format=None):
        """
        Create a new data channel with data set by X (input) and Y (output).


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
            raise Exception("X and Y must be lists or numpy arrays")

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise Exception("X must be either a one or two dimensional array of data")
        if Y.ndim != 1:
            raise Exception("Y must be a one dimensional array of data")
        if X.shape[0] != Y.shape[0]:
            raise Exception("X and Y must be of the same length")
        
        n = X.shape[0]
        input_dims = X.shape[1]

        if format == None:
            format = [FormatNumber] * input_dims
        if not isinstance(format, list):
            format = [format]
        if len(format) != input_dims:
            raise Exception("format must be defined for all input dimensions")

        # sort on X for single input dimensions
        if input_dims == 1:
            ind = np.argsort(X, axis=0)
            X = np.take_along_axis(X, ind, axis=0)
            Y = np.take_along_axis(Y, ind[:,0], axis=0)
        
        self.name = name
        self.X = X # shape (n, input_dims)
        self.Y = Y # shape (n)
        self.mask = np.array([True] * n)
        self.F = None
        self.X_pred = np.array([])
        self.Y_mu_pred = np.array([])
        self.Y_var_pred = np.array([])

        self.input_label = [''] * input_dims
        self.output_label = ''
        self.formatters = format
        self.transformations = []

    def __str__(self):
        return "x=%s; y=%s" % (self.X.tolist(), self.Y.tolist())

    def _encode(self):
        F = self.F
        if F != None:
            F = str(dill.dumps(F))

        return {
            'name': self.name,
            'X': self.X,
            'Y': self.Y,
            'mask': self.mask,
            'F': F,
            'X_pred': X_pred,
            'Y_mu_pred': Y_mu_pred,
            'Y_var_pred': Y_var_pred,
            'input_label': self.input_label,
            'output_label': self.output_label,
            'formatters': self.formatters,
            'transformations': self.transformations,
        }

    def _decode(d):
        self = Data()
        self.name = d['name']
        self.X = d['X']
        self.Y = d['Y']
        self.mask = d['mask']
        self.X_pred = d['X_pred']
        self.Y_mu_pred = d['Y_mu_pred']
        self.Y_var_pred = d['Y_var_pred']
        self.input_label = d['input_label']
        self.output_label = d['output_label']
        self.formatters = d['formatters']
        self.transformations = d['transformations']

        F = d['F']
        if F != None:
            F = dill.loads(eval(F))
        self.F = F

        return self

    def set_name(self, name):
        self.name = name

    def set_labels(self, input, output):
        if isinstance(input, str):
            input = [input]
        elif not isinstance(input, list) or not all(isinstance(item, str) for item in input):
            raise Exception("input labels must be list of strings")
        if not isinstance(output, str):
            raise Exception("output label must be string")
        if len(input) != self.get_input_dims():
            raise Exception("input labels must have the same input dimensions as the data")

        self.input_label = input
        self.output_label = output

    def set_function(self, f):
        """
        Sets a (latent) function corresponding to the channel. The function must take one parameter X (shape (n,input_dims) and output Y (shape (n)).

        This is used for plotting functionality and is optional.
        """
        _check_function(f, self.get_input_dims())
        self.F = f

    def copy(self):
        return copy.deepcopy(self)

    def transform(self, transformer):
        t = transformer
        if '__init__' in vars(transformer):
            t = transformer(self)

        self.Y = t.forward(self.X, self.Y)
        if self.F != None:
            f = self.F
            self.F = lambda x: t.forward(x, f(x))
    
    def filter(self, start, end):
        if self.get_input_dims() != 1:
            raise Exception("can only filter on one dimensional input data")
        
        cstart = self.formatters[0].parse(start)
        cend = self.formatters[0].parse(end)
        ind = (self.X[:,0] >= cstart) & (self.X[:,0] < cend)

        self.X = np.expand_dims(self.X[ind,0], 1)
        self.Y = self.Y[ind]
        self.mask = self.mask[ind]

    def aggregate(self, duration, f=np.mean):
        if self.get_input_dims() != 1:
            raise Exception("can only aggregate on one dimensional input data")
        
        start = self.X[0,0]
        end = self.X[-1,0]
        step = self.formatters[0].parse_duration(duration)

        X = np.arange(start+step/2, end+step/2, step)
        Y = np.empty((len(X)))
        for i in range(len(X)):
            ind = (self.X[:,0] >= X[i]-step/2) & (self.X[:,0] < X[i]+step/2)
            Y[i] = f(self.Y[ind])

        self.X = np.expand_dims(X, 1)
        self.Y = Y
        self.mask = np.array([True] * len(self.X))

    ################################################################

    def has_removed_obs(self):
        return False in self.mask

    def get_input_dims(self):
        """
        Returns the input dimensions, where 
        length of the second dimension for X and Y when using add().
        """
        return self.X.shape[1]

    def get_obs(self):
        """
        Returns the observations for a given channel.
        """
        return self.X[self.mask,:], self.Y[self.mask]
    
    def get_all_obs(self):
        """
        Returns all observations (including removed observations) for a given channel.
        """
        return self.X, self.Y

    def get_del_obs(self):
        """
        Returns the removed observations for a given channel.
        """
        return self.X[~self.mask,:], self.Y[~self.mask]

    ################################################################
    
    def remove_randomly(self, n=None, pct=None):
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
        if n == None:
            if pct == None:
                n = 0
            else:
                n = int((1-pct) * self.X.shape[0])

        idx = np.random.choice(self.X.shape[0], n, replace=False)
        self.mask[idx] = False
    
    def remove_range(self, start=None, end=None):
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

        if start == None:
            start = np.min(self.X[:,0])
        else:
            start = self.formatters[0].parse(start)
        if end == None:
            end = np.max(self.X[:,0])
        else:
            end = self.formatters[0].parse(end)

        idx = np.where(np.logical_and(self.X[:,0] >= start, self.X[:,0] <= end))
        self.mask[idx] = False
    
    def remove_rel_range(self, start, end):
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
            raise exception("can only remove ranges on one dimensional input data")

        start = self.formatters[0].parse(start)
        end = self.formatters[0].parse(end)

        x_min = np.min(self.X[:,0])
        x_max = np.max(self.X[:,0])
        start = x_min + np.round(max(0.0, min(1.0, start)) * (x_max-x_min))
        end = x_min + np.round(max(0.0, min(1.0, end)) * (x_max-x_min))

        idx = np.where(np.logical_and(self.X[:,0] >= start, self.X[:,0] <= end))
        self.mask[idx] = False

    def remove_random_ranges(self, n, size):
        """
        Removes a number of ranges on a channel. Makes only sense if your input X is sorted.

        Args:
            channel (str, int): Channel to set prediction, can be either a string with the name
                of the channel or a integer with the index.

            n (int): Number of ranges to remove.

            size (int): Width of ranges to remove.
        """
        if n < 1 or size < 1:
            return

        m = self.X.shape[0] - n*size
        if m <= 0:
            raise Exception("no data left after removing ranges")

        locs = np.round(np.sort(np.random.rand(n)) * m)
        for i in range(len(locs)):
            loc = int(locs[i] + i * size)
            idx = np.arange(loc, loc+size)
            self.mask[idx] = False
    
    ################################################################

    def set_pred_range(self, start=None, end=None, n=None, step=None):
        """
        Sets the prediction range for a certain channel in the interval [start,end].
        with either a stepsize step or a number of points n.

        Args:
            channels (str, int, list, '*'): Channel to set prediction, can be either a string with the name
                of the channel or a integer with the index.

            start (float, optional): Initial value of range, if not passed the first point of training
                data is taken. Default to None.

            end (float, optional): Final value of range, if not passed the last point of training
                data is taken. Default to None.

            step (float, optional): Spacing between values.

            n (int, optional): Number of samples to generate.

            If neither "step" or "n" is passed, default number of points is 100.
        """
        if self.get_input_dims() != 1:
            raise exception("can only set prediction range on one dimensional input data")

        cstart = start
        if cstart == None:
            cstart = self.X[0,:]
        elif isinstance(cstart, list):
            for i in range(self.get_input_dims()):
                cstart[i] = self.formatters[i].parse(cstart[i])
        else:
            cstart = self.formatters[0].parse(cstart)

        cend = end
        if cend == None:
            cend = self.X[-1,:]
        elif isinstance(cend, list):
            for i in range(self.get_input_dims()):
                cend[i] = self.formatters[i].parse(cend[i])
        else:
            cend = self.formatters[0].parse(cend)
        
        cstart = _normalize_input_dims(cstart, self.get_input_dims())
        cend = _normalize_input_dims(cend, self.get_input_dims())

        # TODO: works for multi input dims?
        if cend <= cstart:
            raise Exception("start must be lower than end")

        # TODO: prediction range for multi input dimension; fix other axes to zero so we can plot?
        if step == None and n != None:
            self.X_pred = np.empty((n, self.get_input_dims()))
            for i in range(self.get_input_dims()):
                self.X_pred[:,i] = np.linspace(cstart[i], cend[i], n)
        else:
            if self.get_input_dims() != 1:
                raise Exception("cannot use step for multi dimensional input, use n")
            cstep = step
            if cstep == None:
                cstep = (cend[0]-cstart[0])/100
            else:
                cstep = self.formatters[0].parse(cstep)
            self.X_pred = np.arange(cstart[0], cend[0]+cstep, cstep).reshape(-1, 1)
    
    def set_pred(self, x):
        """
        Sets the prediction range using a list of Numpy array for a certain channel.

        Args:
            channel (str, int): Channel to set prediction, can be either a string with the name
                of the channel or a integer with the index.
            x (ndarray): Numpy array with input values for channel.
        """
        if x.ndim != 2 or x.shape[1] != self.get_input_dims():
            raise Exception("x shape must be (n,input_dims)")
        if isinstance(x, list):
            x = np.array(x)
        elif not isinstance(x, np.ndarray):
            raise Exception("x expected to be a list or Numpy array")

        self.X_pred = x

    ################################################################

    def get_nyquist_estimation(self):
        """
        Estimate nyquist frequency for each channel by taking
        0.5/min dist of points.

        Returns:
            Numpy array of length equal to number of channels.
        """
        input_dims = self.get_input_dims()

        nyquist = np.empty((input_dims))
        for i in range(self.get_input_dims()):
            x = np.sort(self.X[:,i])
            dist = np.abs(x[1:]-x[:-1]) # TODO: assumes X is sorted, use average distance instead of minimal distance?
            dist = np.min(dist[np.nonzero(dist)])
            nyquist[i] = 0.5/dist
        return nyquist

    def get_bnse_estimation(self, Q=1):
        """
        Peaks estimation using BNSE (Bayesian Non-parametric Spectral Estimation)

        Returns:
            freqs, amps: Each one is a input_dim x n_channels x Q array with 
                the frequency values and amplitudes of the peaks.

        """
        input_dims = self.get_input_dims()

        freqs = np.zeros((input_dims, Q))
        amps = np.zeros((input_dims, Q))

        nyquist = self.get_nyquist_estimation()
        for i in range(input_dims):
            x = self.X[:,i]
            y = self.Y
            bnse = bse(x, y)
            bnse.set_freqspace(nyquist[i], dimension=5000)
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
            
            freqs[i,:] = 2*np.pi*peaks
            amps[i,:] = amplitudes
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

        freqs = np.zeros((input_dims, Q))
        amps = np.zeros((input_dims, Q))

        nyquist = self.get_nyquist_estimation() * 2 * np.pi
        for i in range(input_dims):
            freq_space = np.linspace(0, nyquist[i], n_ls+1)[1:]
            pgram = lombscargle(self.X[:,i], self.Y, freq_space)
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

            freqs[i,:] = peaks[:,1]
            amps[i,:] = peaks[:,0]

        return freqs / np.pi / 2, amps
    
    def get_gm_estimation(self):
        # TODO: use sklearn.mixture.GaussianMixture to retrieve fitted gaussian mixtures to spectral data
        pass

    def plot(self, title=None, filename=None, show=True):
        # TODO: ability to plot conditional or marginal distribution to reduce input dims
        if self.get_input_dims() > 2:
            raise Exception("cannot plot more than two input dimensions")
        if self.get_input_dims() == 2:
            raise Exception("two dimensional input data not yet implemented") # TODO

        sns.set(font_scale=2)
        sns.axes_style("darkgrid")
        sns.set_style("whitegrid")

        fig, axes = plt.subplots(1, 1, figsize=(20, 5), constrained_layout=True, squeeze=False)
        if title != None:
            fig.suptitle(title, fontsize=36)

        plotting_pred = False
        plotting_F = False
        plotting_obs = False

        if self.Y_mu_pred.size != 0:
            lower = self.Y_mu_pred - self.Y_var_pred
            upper = self.Y_mu_pred + self.Y_var_pred
            axes[0,0].plot(self.X_pred[:,0], self.Y_mu_pred, 'b-', lw=3)
            axes[0,0].fill_between(self.X_pred[:,0], lower, upper, color='b', alpha=0.1)
            axes[0,0].plot(self.X_pred[:,0], lower, 'b-', lw=1, alpha=0.5)
            axes[0,0].plot(self.X_pred[:,0], upper, 'b-', lw=1, alpha=0.5)
            plotting_pred = True

        if self.F != None:
            n = len(self.X[:,0])*10
            x_min = np.min(self.X[:,0])
            x_max = np.max(self.X[:,0])

            x = np.empty((n, 1))
            x[:,0] = np.linspace(x_min, x_max, n)
            y = self.F(x)

            axes[0,0].plot(x[:,0], y, 'r--', lw=1)
            plotting_F = True

        axes[0,0].plot(self.X[:,0], self.Y, 'k-')

        if self.has_removed_obs():
            X, Y = self.get_obs()
            axes[0,0].plot(X[:,0], Y, 'k.', mew=2, ms=8)
            plotting_obs = True

        axes[0,0].set_xlabel(self.input_label[0])
        axes[0,0].set_ylabel(self.output_label)
        axes[0,0].set_title(self.name, fontsize=30)
        formatter = matplotlib.ticker.FuncFormatter(lambda x,pos: self.formatters[0].format(x))
        axes[0,0].xaxis.set_major_formatter(formatter)

        # build legend
        if plotting_F or plotting_obs:
            legend = []
            legend.append(plt.Line2D([0], [0], ls='-', color='k', label='Data'))
            if plotting_F:
                legend.append(plt.Line2D([0], [0], ls='--', color='r', label='Latent function'))
            if plotting_obs:
                legend.append(plt.Line2D([0], [0], ls='', marker='.', color='k', mew=2, ms=8, label='Training'))
            if plotting_pred:
                legend.append(plt.Line2D([0], [0], ls='-', color='b', lw=3, label='Prediction'))
            plt.legend(handles=legend, loc='best')

        if filename != None:
            plt.savefig(filename+'.pdf', dpi=300)
        if show:
            plt.show()

    def plot_spectrum(self, method='lombscargle', title=None, angular=False, per=None, maxfreq=None, filename=None, show=True):
        # TODO: ability to plot conditional or marginal distribution to reduce input dims
        if self.get_input_dims() > 2:
            raise Exception("cannot plot more than two input dimensions")
        if self.get_input_dims() == 2:
            raise Exception("two dimensional input data not yet implemented") # TODO

        sns.set(font_scale=2)
        sns.axes_style("darkgrid")
        sns.set_style("whitegrid")

        fig, axes = plt.subplots(1, 1, figsize=(20, 5), constrained_layout=True, squeeze=False)
        if title != None:
            fig.suptitle(title, fontsize=36)

        X_space = self.X[:,0].copy()

        formatter = self.formatters[0]
        factor, name = formatter.scale(per)
        if name != None:
            axes[0,0].set_xlabel('Frequency (1/'+name+')')
        else:
            axes[0,0].set_xlabel('Frequency')

        if not angular:
            X_space *= 2 * np.pi
        X_space /= factor

        freq = maxfreq
        if freq == None:
            dist = np.abs(X_space[1:]-X_space[:-1])
            freq = 1/np.average(dist)

        X = np.linspace(0.0, freq, 10001)[1:]
        if method == 'lombscargle':
            Y = lombscargle(X_space, self.Y, X)
        else:
            raise Exception('Periodogram method "%s" does not exist' % (method))

        axes[0,0].plot(X, Y, 'k-')
        axes[0,0].set_title(self.name + ' spectrum', fontsize=30)
        axes[0,0].set_yticks([])
        axes[0,0].set_ylim(0, None)

        if filename != None:
            plt.savefig(filename+'.pdf', dpi=300)
        if show:
            plt.show()

def _check_function(f, input_dims):
    if not inspect.isfunction(f):
        raise Exception("F must be a function taking X as a parameter")

    sig = inspect.signature(f)
    if not len(sig.parameters) == 1:
        raise Exception("F must be a function taking X as a parameter")

    x = np.ones((1, input_dims))
    y = f(x)
    if len(y.shape) != 1 or y.shape[0] != 1:
        raise Exception("F must return Y with shape (n), note that X has shape (n,input_dims)")

def _normalize_input_dims(x, input_dims):
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
    if input_dims != None and len(x) != input_dims:
        raise Exception("input must be a scalar for single-dimension input or a list of values for each input dimension")
    return x
