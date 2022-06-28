import re
import copy
import inspect
import datetime
import logging
import collections

import numpy as np
import pandas as pd
from scipy import signal
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pandas.plotting import register_matplotlib_converters

from .transformer import Transformer
from .init import BNSE
from .util import plot_spectrum

register_matplotlib_converters()

logger = logging.getLogger('mogptk')

def LoadSplitData(x_train, x_test, y_train, y_test, name=""):
    """
    Load from a split data set.

    Args:
        x_train (numpy.ndarray): Training input of shape (data_points,input_dims).
        x_test (numpy.ndarray): Testing input of shape (test_points,input_dims).
        y_train (numpy.ndarray): Training output of shape (data_points,).
        y_test (numpy.ndarray): Testing output of shape (test_points,).
        name (str): Name of data.

    Returns:
        mogptk.data.Data

    Examples:
        >>> LoadSplitData(x_train, x_test, y_train, y_test, name='Sine wave')
        <mogptk.data.Data at ...>
    """
    if not isinstance(x_train, np.ndarray):
        x_train = np.array(x_train)
    if not isinstance(x_test, np.ndarray):
        x_test = np.array(x_test)
    if not isinstance(y_train, np.ndarray):
        y_train = np.array(y_train)
    if not isinstance(y_test, np.ndarray):
        y_test = np.array(y_test)
    if x_train.ndim == 1:
        x_train = x_train.reshape(-1,1)
    if x_test.ndim == 1:
        x_test = x_test.reshape(-1,1)
    if y_train.ndim == 2 and y_train.shape[1] == 1:
        y_train = y_train.reshape(-1)
    if y_test.ndim == 2 and y_test.shape[1] == 1:
        y_test = y_test.reshape(-1)

    if x_train.ndim != 2 or x_test.ndim != 2:
        raise ValueError("x data must have shape (data_points,input_dims)")
    if y_train.ndim != 1 or y_test.ndim != 1:
        raise ValueError("y data must have shape (data_points,)")
    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError("x_train and y_train must have the same number of data points")
    if x_test.shape[0] != y_test.shape[0]:
        raise ValueError("x_test and y_test must have the same number of data points")
    if x_train.shape[1] != x_test.shape[1]:
        raise ValueError("x_train and x_test must have the same number of input dimensions")

    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    test_indices = np.arange(len(x_train),len(x))

    data = Data(x, y, name=name)
    data.remove_indices(test_indices)
    return data

def LoadFunction(f, start, end, n, var=0.0, name="", random=False):
    """
    LoadFunction loads a dataset from a given function y = f(x) + Normal(0,var). It will pick `n` data points between start and end for the X axis for which `f` is being evaluated. By default the `n` points are spread equally over the interval, with `random=True` they will be picked randomly.

    Args:
        f (function): Function taking a numpy.ndarray X of shape (data_points,) for each input dimension and returning a numpy.ndarray Y of shape (data_points,).
        start (float, list): Define start of interval.
        end (float, list): Define end of interval.
        n (int, list): Number of data points to pick between start and end.
        var (float): Variance added to the output.
        name (str): Name of data.
        random (boolean): Select points randomly between start and end.

    Returns:
        mogptk.data.Data

    Examples:
        >>> LoadFunction(lambda x,y: np.sin(3*x)+np.cos(2*y), 0, 10, n=200, var=0.1, name='Sine wave')
        <mogptk.data.Data at ...>
    """

    if isinstance(start, np.ndarray):
        if start.ndim == 0:
            start = [start.item()]
        else:
            start = list(start)
    elif _is_iterable(start):
        start = list(start)
    else:
        start = [start]
    if isinstance(end, np.ndarray):
        if end.ndim == 0:
            end = [end.item()]
        else:
            end = list(end)
    elif _is_iterable(end):
        end = list(end)
    else:
        end = [end]
    if type(start[0]) is not type(end[0]):
        raise ValueError("start and end must be of the same type")
    if len(start) != len(end):
        raise ValueError("start and end must be of the same length")

    input_dims = len(start)
    for i in range(input_dims):
        if not _is_homogeneous_type([start[i] + end[i]]):
            raise ValueError("start and end must have elements of the same type")

        if isinstance(start[i], datetime.datetime) or isinstance(start[i], str) or isinstance(start[i], np.datetime64):
            try:
                start[i] = np.datetime64(start[i], 'us')
                end[i] = np.datetime64(end[i], 'us')
            except:
                raise ValueError("start and end must have matching number or datetime data type")
        else:
            try:
                start[i] = np.float64(start[i])
                end[i] = np.float64(end[i])
            except:
                raise ValueError("start and end must have matching number or datetime data type")

    _check_function(f, input_dims, [isinstance(start[i], np.datetime64) for i in range(input_dims)])

    if _is_iterable(n):
        n = list(n)
    else:
        n = [n] * input_dims
    if len(n) != input_dims:
        raise ValueError("n must be a scalar or a list of values for each input dimension")
    if _is_iterable(random):
        random = list(random)
    else:
        random = [random] * input_dims
    if len(random) != input_dims:
        raise ValueError("random must be a scalar or a list of values for each input dimension")

    for i in range(input_dims):
        if random[i] and isinstance(start[i], np.datetime64):
            if input_dims == 1:
                raise ValueError("cannot use random for datetime inputs for input dimension %d", (i,))
            else:
                raise ValueError("cannot use random for datetime inputs")

    x = [None] * input_dims
    for i in range(input_dims):
        if start[i] >= end[i]:
            if input_dims == 1:
                raise ValueError("start must be lower than end")
            else:
                raise ValueError("start must be lower than end for input dimension %d" % (i,))

        if isinstance(start[i], np.datetime64):
            dt = (end[i]-start[i]) / float(n[i]-1)
            dt = _timedelta64_to_higher_unit(dt)
            x[i] = np.arange(start[i], start[i]+dt*(n[i]-1)+np.timedelta64(1,'us'), dt, dtype=start[i].dtype)
        elif random[i]:
            x[i] = np.random.uniform(start[i], end[i], n[i])
        else:
            x[i] = np.linspace(start[i], end[i], n[i])

        N_tile = np.prod(n[:i])
        N_repeat = np.prod(n[i+1:])
        x[i] = np.tile(np.repeat(x[i], N_repeat), N_tile)

    y = f(*x)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:,0]
    N = np.prod(n)
    y += np.random.normal(0.0, var, (N,))

    data = Data(x, y, name=name)
    data.set_function(f)
    return data

################################################################
################################################################
################################################################

class Data:
    def __init__(self, X, Y, Y_err=None, name=None, x_labels=None, y_label=None):
        """
        Data class that holds all observations, latent functions and predicted data.

        This class accepts the data directly, otherwise you can load data conveniently using `LoadFunction`, `LoadCSV`, `LoadDataFrame`, etc. The data class allows to modify the data before passing into the model. Examples are transforming data, such as detrending or taking the log, removing data ranges to simulate sensor failure, and aggregating data for given spans on X, such as aggregating daily data into weekly data. Additionally, we also use this class to set the range we want to predict.

        It is possible to use the format given by `numpy.meshgrid` for X as a list of numpy arrays for each input dimension, and its values in Y. Each input dimension and Y must have shape (N1,N2,...,Nn) where n is the number of input dimensions and N the number of data points per input dimension.

        Args:
            X (list, numpy.ndarray, pandas.Series, dict): Independent variable data of shape (data_points,) or (data_points,input_dims), or a list with elements of shape (data_points,) for each input dimension.
            Y (list, numpy.ndarray, pandas.Series): Dependent variable data of shape (data_points,).
            Y_err (list, numpy.ndarray, pandas.Series): Standard deviation of the dependent variable data of shape (n,).
            name (str): Name of data.
            x_labels (str, list of str): Name or names of input dimensions.
            y_label (str): Name of output dimension.

        Examples:
            >>> channel = mogptk.Data([0, 1, 2, 3], [4, 3, 5, 6])
        """

        # convert dicts to lists
        if x_labels is not None:
            if isinstance(x_labels, str):
                x_labels = [x_labels]
            if not isinstance(x_labels, list) or not all(isinstance(label, str) for label in x_labels):
                raise ValueError("x_labels must be a string or list of strings for each input dimension")

            if isinstance(X, dict):
                it = iter(X.values())
                first = len(next(it))
                if not all(isinstance(x, (list, np.ndarray)) for x in X.values()) or not all(len(x) == first for x in it):
                    raise ValueError("X dict should contain all lists or numpy.ndarrays where each has the same length")
                if not all(key in X for key in x_labels):
                    raise ValueError("X dict must contain all keys listed in x_labels")
                X = [X[key] for key in x_labels]

        X, X_dtypes = self._format_X(X)
        Y = self._format_Y(Y)
        if Y_err is not None:
            Y_err = self._format_Y(Y_err)

        # convert meshgrids to flat arrays
        if 1 < X[0].ndim and 1 < Y.ndim and X[0].shape == Y.shape:
            X = [np.ravel(x) for x in X]
            Y = np.ravel(Y)
            if Y_err is not None:
                Y_err = np.ravel(Y_err)

        if X.ndim != 2:
            raise ValueError("X must have shape (data_points,input_dims)")
        if Y.ndim != 1:
            raise ValueError("Y must have shape (data_points,)")
        if Y.shape[0] == 0:
            raise ValueError("X and Y must have a length greater than zero")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must be of the same length")
        if Y_err is not None and Y.shape != Y_err.shape:
            raise ValueError("Y and Y_err must have the same shape")

        self.X = X # shape (n,input_dims)
        self.Y = Y # shape (n)
        self.Y_err = Y_err # shape (n) or None
        self.X_pred = None
        self.mask = np.array([True] * Y.shape[0])
        self.F = None

        self.X_dtypes = X_dtypes
        self.Y_transformer = Transformer()

        input_dims = X.shape[1]
        self.removed_ranges = [[]] * input_dims
        self.X_labels = ['X'] * input_dims
        if 1 < input_dims:
            for i in range(input_dims):
                self.X_labels[i] = 'X%d' % (i,)
        if isinstance(x_labels, list) and all(isinstance(item, str) for item in x_labels):
            self.X_labels = x_labels

        self.name = None
        if isinstance(name, str):
            self.name = name
        elif isinstance(y_label, str):
            self.name = y_label

        self.Y_label = 'Y'
        if isinstance(y_label, str):
            self.Y_label = y_label

    def _format_X(self, X):
        if isinstance(X, list) and 0 < len(X):
            islist = False
            if all(isinstance(x, list) for x in X):
                islist = True
                m = len(X[0])
                if not all(len(x) == m for x in X[1:]):
                    raise ValueError("X list items must all be lists of the same length")
                if not all(all(isinstance(val, (int, float, datetime.datetime, np.datetime64)) for val in x) for x in X):
                    raise ValueError("X list items must all be lists of numbers or datetime")
                if not all(_is_homogeneous_type(x) for x in X):
                    raise ValueError("X list items must all be lists with elements of the same type")
            elif all(isinstance(x, np.ndarray) for x in X):
                islist = True
                m = len(X[0])
                if not all(len(x) == m for x in X[1:]):
                    raise ValueError("X list items must all be numpy.ndarrays of the same length")
            elif not all(isinstance(x, (int, float, datetime.datetime, np.datetime64)) for x in X):
                raise ValueError("X list items must be all lists, all numpy.ndarrays, or all numbers or datetime")
            elif not _is_homogeneous_type(X):
                raise ValueError("X list items must all have elements of the same type")

            if islist:
                X = [np.array(x) for x in X]
            else:
                X = [np.array(X)]
        elif isinstance(X, (np.ndarray, pd.Series)):
            if isinstance(X, pd.Series):
                X = X.to_numpy()
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            if X.ndim != 2:
                raise ValueError("X must be either a one or two dimensional array of data")
            X = [X[:,i] for i in range(X.shape[1])]
        else:
            raise ValueError("X must be list, numpy.ndarray, or pandas.Series")

        # try to cast unknown data types, X becomes np.float64 or np.datetime64
        input_dims = len(X)
        if hasattr(self, 'X_dtypes'):
            for i in range(input_dims):
                try:
                    X[i] = X[i].astype(self.X_dtypes[i])
                except:
                    raise ValueError("X data must have valid data types for each input dimension")
        else:
            for i in range(input_dims):
                if X[i].dtype == np.object_ or np.issubdtype(X[i].dtype, np.character):
                    # convert datetime.datetime or strings to np.datetime64
                    try:
                        X[i] = X[i].astype(np.datetime64)
                    except:
                        raise ValueError("X data must have a number or datetime data type")
                elif not np.issubdtype(X[i].dtype, np.datetime64):
                    try:
                        X[i] = X[i].astype(np.float64)
                    except:
                        raise ValueError("X data must have a number or datetime data type")

                # convert X datetime64[us] to a higher unit like s, m, h, D, ...
                if np.issubdtype(X[i].dtype, np.datetime64):
                    X[i] = _datetime64_to_higher_unit(X[i])

        dtypes = [x.dtype for x in X]
        X = np.array([x.astype(np.float64) for x in X]).T
        if X.size == 0:
            raise ValueError("X data must not be empty")
        if not np.isfinite(X).all():
            raise ValueError("X data must not contains NaNs or infinities")
        return X, dtypes # shape (n,input_dims)

    def _format_Y(self, Y):
        if isinstance(Y, list):
            if not all(isinstance(y, (int, float)) for y in Y):
                raise ValueError("Y list items must all be numbers")
            elif not _is_homogeneous_type(Y):
                raise ValueError("Y list items must all have elements of the same type")
            Y = np.array(Y)
        elif isinstance(Y, pd.Series):
            Y = Y.to_numpy()
        elif not isinstance(Y, np.ndarray):
            raise ValueError("Y must be list, numpy.ndarray, or pandas.Series")

        # try to cast unknown data types, Y becomes np.float64
        try:
            Y = Y.astype(np.float64)
        except:
            raise ValueError("Y data must have a number data type")

        if Y.ndim == 2 and Y.shape[1] == 1:
            Y = Y.reshape(-1)

        if Y.shape[0] == 0:
            raise ValueError("Y data must not be empty")
        if not np.isfinite(Y).all():
            raise ValueError("Y data must not contains NaNs or infinities")
        return Y # shape (n,)

    def __repr__(self):
        df = pd.DataFrame()
        for i in range(self.X.shape[1]):
            df[self.X_labels[i]] = self.X[:,i]
        df[self.Y_label] = self.Y
        return repr(df)

    def copy(self):
        """
        Make a deep copy of `Data`.

        Returns:
            mogptk.data.Data

        Examples:
            >>> other = data.copy()
        """
        return copy.deepcopy(self)

    def set_name(self, name):
        """
        Set name for data channel.

        Args:
            name (str): Name of data.

        Examples:
            >>> data.set_name('Channel A')
        """
        self.name = name

    def set_labels(self, x_labels, y_label):
        """
        Set axis labels for plots.

        Args:
            x_labels (str, list of str): X data names for each input dimension.
            y_label (str): Y data name for output dimension.

        Examples:
            >>> data.set_labels(['X', 'Y'], 'Cd')
        """
        if isinstance(x_labels, str):
            x_labels = [x_labels]
        elif not isinstance(x_labels, list) or not all(isinstance(item, str) for item in x_labels):
            raise ValueError("x_labels must be list of strings")
        if not isinstance(y_label, str):
            raise ValueError("y_label must be string")
        if len(x_labels) != self.get_input_dims():
            raise ValueError("x_labels must have the same input dimensions as the data")

        self.X_labels = x_labels
        self.Y_label = y_label

    def set_function(self, f):
        """
        Set the (latent) function for the data, ie. the theoretical or true signal. This is used for plotting purposes and is optional.
    
        Args:
        f (function): Function taking a numpy.ndarray X of shape (data_points,) for each input dimension and returning a numpy.ndarray Y of shape (data_points,).

        Examples:
            >>> data.set_function(lambda x,y: np.sin(3*x)+np.cos(2*y))
        """
        _check_function(f, self.get_input_dims(), [_is_datetime64(self.X_dtypes[i]) for i in range(self.get_input_dims())])
        self.F = f

    def transform(self, transformer):
        """
        Transform the Y axis data by using one of the provided transformers, such as `TransformDetrend`, `TransformLinear`, `TransformLog`, `TransformNormalize`, `TransformStandard`, etc.

        Args:
            transformer (obj): Transformer object derived from TransformBase.

        Examples:
            >>> data.transform(mogptk.TransformDetrend(degree=2))        # remove polynomial trend
            >>> data.transform(mogptk.TransformLinear(slope=1, bias=2))  # remove linear trend
            >>> data.transform(mogptk.TransformLog)                      # log transform the data
            >>> data.transform(mogptk.TransformNormalize)                # transform to [-1,1]
            >>> data.transform(mogptk.TransformStandard)                 # transform to mean=0, var=1
        """
        self.Y_transformer.append(transformer, self.Y, self.X)
    
    def filter(self, start, end, dim=None):
        """
        Filter the data range to be between `start` and `end` in the X axis.

        Args:
            start (float, str, list): Start of interval.
            end (float, str, list): End of interval (not included).
            dim (int): Input dimension to apply to, if not specified applies to all input dimensions.

        Examples:
            >>> data.filter(3, 8)
        
            >>> data.filter('2016-01-15', '2016-06-15')
        """
        start = self._normalize_x_val(start)
        end = self._normalize_x_val(end)
        
        if dim is not None:
            ind = np.logical_and(self.X[:,dim] >= start[dim], self.X[:,dim] < end[dim])
        else:
            ind = np.logical_and(self.X[:,0] >= start[0], self.X[:,0] < end[0])
            for i in range(1,self.get_input_dims()):
                ind = np.logical_and(ind, np.logical_and(self.X[:,i] >= start[i], self.X[:,i] < end[i]))

        self.X = self.X[ind,:]
        self.Y = self.Y[ind]
        if self.Y_err is not None:
            self.Y_err = self.Y_err[ind]
        self.mask = self.mask[ind]

    def aggregate(self, duration, f=np.mean, f_err=None):
        """
        Aggregate the data by duration and apply a function to obtain a reduced dataset.

        For example, group daily data by week and take the mean. The duration can be set as a number which defined the intervals on the X axis, or by a string written in the duration format in case the X axis has data type `numpy.datetime64`. The duration format uses: Y=year, M=month, W=week, D=day, h=hour, m=minute, and s=second. For example, 3W1D means three weeks and one day, ie. 22 days, or 6M to mean six months.

        Args:
            duration (float, str): Duration along the X axis or as a string in the duration format.
            f (function): Function to use to reduce data mapping a numpy array to a scalar, such as `numpy.mean`.
            f_err (function): Function to use to reduce data mapping a numpy array to a scalar, such as `numpy.mean`. This function is used to map the Y_err error values and uses by default the same function as for the Y values.

        Examples:
            >>> data.aggregate(5)

            >>> data.aggregate('2W', f=np.sum)
        """
        if 1 < self.get_input_dims():
            raise ValueError("aggregate works only with a single input dimension")

        start = np.min(self.X[:,0])
        end = np.max(self.X[:,0])
        step = _parse_delta(duration, self.X_dtypes[0])
        if f_err is None:
            f_err = f

        X = np.arange(start+step/2, end+step/2, step).reshape(-1,1)
        Y = np.empty((X.shape[0],))
        if self.Y_err is not None:
            Y_err = np.empty((X.shape[0],))
        for i in range(X.shape[0]):
            ind = (self.X[:,0] >= X[i,0]-step/2) & (self.X[:,0] < X[i,0]+step/2)
            Y[i] = f(self.Y[ind])
            if self.Y_err is not None:
                Y_err[i] = f_err(self.Y_err[ind])
        self.X = X
        self.Y = Y
        if self.Y_err is not None:
            self.Y_err = Y_err
        self.mask = np.array([True] * len(self.Y))

    ################################################################

    def get_name(self):
        """
        Return the name of the channel.

        Returns:
            str

        Examples:
            >>> data.get_name()
            'A'
        """
        return self.name

    def has_test_data(self):
        """
        Returns True if observations have been removed using the `remove_*` methods.

        Returns:
            boolean

        Examples:
            >>> data.has_test_data()
            True
        """
        return False in self.mask

    def get_input_dims(self):
        """
        Returns the number of input dimensions.

        Returns:
            int: Number of input dimensions.

        Examples:
            >>> data.get_input_dims()
            2
        """
        return self.X.shape[1]
    
    def get_data(self, transformed=False):
        """
        Returns all observations, train and test.

        Arguments:
            transformed (boolean): Return transformed data.

        Returns:
            numpy.ndarray: X data of shape (data_points,input_dims).
            numpy.ndarray: Y data of shape (data_points,).

        Examples:
            >>> x, y = data.get_data()
        """
        if transformed:
            return self.X, self.Y_transformer.forward(self.Y, self.X)
        return self.X, self.Y

    def get_train_data(self, transformed=False):
        """
        Returns the observations used for training.

        Arguments:
            float64 (boolean): Return as float64s.
            transformed (boolean): Return transformed data.

        Returns:
            numpy.ndarray: X data of shape (data_points,input_dims).
            numpy.ndarray: Y data of shape (data_points,).

        Examples:
            >>> x, y = data.get_train_data()
        """
        if transformed:
            return self.X[self.mask,:], self.Y_transformer.forward(self.Y[self.mask], self.X[self.mask,:])
        return self.X[self.mask,:], self.Y[self.mask]

    def get_test_data(self, transformed=False):
        """
        Returns the observations used for testing which correspond to the removed points.

        Arguments:
            transformed (boolean): Return transformed data.

        Returns:
            list of numpy.ndarray: X data of shape [(n,)] * input_dims.
            numpy.ndarray: Y data of shape (n,).

        Examples:
            >>> x, y = data.get_test_data()
        """
        X = self.X[~self.mask,:]
        if self.F is not None:
            if X.shape[0] == 0:
                X, _ = self.get_data()
            Y = self.F(X)
            if transformed:
                Y = self.Y_transformer.forward(Y, X)
            return X, Y
        if transformed:
            return X, self.Y_transformer.forward(self.Y[~self.mask], X)
        return X, self.Y[~self.mask]

    ################################################################

    def reset(self):
        """
        Reset the dataset and undo the removal of data points. That is, this reverts any calls to `remove_randomly`, `remove_range`, `remove_relative_range`, `remove_random_ranges`, and `remove_index`. Additionally, also resets the prediction range to the original dataset.
        """
        self.mask[:] = True
        for i in range(len(self.removed_ranges)):
            self.removed_ranges[i] = []
        self.X_pred = None
    
    def remove_randomly(self, n=None, pct=None):
        """
        Removes observations randomly on the whole range. Either `n` observations are removed, or a percentage of the observations.

        Args:
            n (int): Number of observations to remove randomly.
            pct (float): Percentage in interval [0,1] of observations to remove randomly.

        Examples:
            >>> data.remove_randomly(50) # remove 50 observations

            >>> data.remove_randomly(pct=0.9) # remove 90% of the observations
        """
        if n is None:
            if pct is None:
                n = 0
            else:
                n = int(pct * len(self.Y))
        elif not isinstance(n, int) or isinstance(n, float) and not n.is_integer():
            raise ValueError('n must be an integer')

        idx = np.random.choice(len(self.Y), n, replace=False)
        self.mask[idx] = False

    def _add_range(self, start, end, dim):
        # assume current ranges are sorted and non-overlapping
        ranges = self.removed_ranges[dim]

        # find index that sorts on start, ie. here we will insert the new range
        idx = 0
        while idx < len(ranges) and ranges[idx][0] < start:
            idx += 1

        # merge previous range if it overlaps with new
        if 0 < idx and start <= ranges[idx-1][1]:
            start = ranges[idx-1][0]
            idx -= 1

        # merge other ranges if they overlap with new
        rem = 0
        for i in range(idx, len(ranges)):
            if end < ranges[i][0]:
                break
            end = max(end, ranges[i][1])
            rem += 1

        self.removed_ranges[dim] = ranges[:idx] + [(start,end)] + ranges[idx+rem:]
    
    def remove_range(self, start=None, end=None, dim=None):
        """
        Removes observations in the interval `[start,end]`.
        
        Args:
            start (float, str): Start of interval (inclusive). Defaults to the first value in observations.
            end (float, str): End of interval (inclusive). Defaults to the last value in observations.
            dim (int): Input dimension to apply to, if not specified applies to all input dimensions.

        Examples:
            >>> data = mogptk.LoadFunction(lambda x: np.sin(3*x[:,0]), 0, 10, n=200, var=0.1, name='Sine wave')
            >>> data.remove_range(3, 8)
        
            >>> data = mogptk.LoadCSV('gold.csv', 'Date', 'Price')
            >>> data.remove_range('2016-01-15', '2016-06-15')
        """
        if start is None:
            start = [np.min(self.X[:,i]) for i in range(self.get_input_dims())]
        if end is None:
            end = [np.max(self.X[:,i]) for i in range(self.get_input_dims())]

        start = self._normalize_x_val(start)
        end = self._normalize_x_val(end)

        if dim is not None:
            mask = np.logical_and(self.X[:,dim] >= start[dim], self.X[:,dim] <= end[dim])
            self._add_range(start[dim], end[dim], dim)
        else:
            mask = np.logical_and(self.X[:,0] >= start[0], self.X[:,0] <= end[0])
            for i in range(1,self.get_input_dims()):
                mask = np.logical_or(mask, np.logical_and(self.X[:,i] >= start[i], self.X[:,i] <= end[i]))
            for i in range(self.get_input_dims()):
                self._add_range(start[i], end[i], i)
        self.mask[mask] = False
    
    def remove_relative_range(self, start=0.0, end=1.0, dim=None):
        """
        Removes observations between `start` and `end` as a percentage of the number of observations. So `0` is the first observation, `0.5` is the middle observation, and `1` is the last observation.

        Args:
            start (float): Start percentage in interval [0,1].
            end (float): End percentage in interval [0,1].
            dim (int): Input dimension to apply to, if not specified applies to all input dimensions.
        """
        start = self._normalize_x_val(start)
        end = self._normalize_x_val(end)

        xmin = [np.min(self.X[:,i]) for i in range(self.get_input_dims())]
        xmax = [np.max(self.X[:,i]) for i in range(self.get_input_dims())]
        for i in range(self.get_input_dims()):
            start[i] = xmin[i] + max(0.0, min(1.0, start[i])) * (xmax[i]-xmin[i])
            end[i] = xmin[i] + max(0.0, min(1.0, end[i])) * (xmax[i]-xmin[i])
        self.remove_range(start, end, dim)

    def remove_random_ranges(self, n, duration, dim=0):
        """
        Removes a number of ranges to simulate sensor failure. May remove fewer ranges if there is no more room to remove a range in the remaining data.

        Args:
            n (int): Number of ranges to remove.
            duration (float, str): Width of ranges to remove, can use a number or the duration format syntax (see aggregate()).
            dim (int): Input dimension to apply to, defaults to the first input dimension.

        Examples:
            >>> data.remove_random_ranges(2, 5) # remove two ranges that are 5 wide in input space

            >>> data.remove_random_ranges(3, '1d') # remove three ranges that are 1 day wide
        """
        if n < 1:
            return

        delta = _parse_delta(duration, self.X.dtypes[dim])
        m = (np.max(self.X[:,dim])-np.min(self.X[:,dim])) - n*delta
        if m <= 0:
            raise ValueError("no data left after removing ranges")

        locs = self.X[:,dim] <= (np.max(self.X[:,dim])-delta)
        locs[sum(locs)] = True # make sure the last data point can be deleted
        for i in range(n):
            if self.X[locs,dim].shape[0] == 0:
                break # range could not be removed, there is no remaining data range of width delta
            x = self.X[locs,dim][np.random.randint(self.X[locs,dim].shape[0])]
            locs[(self.X[:,dim] > x-delta) & (self.X[:,dim] < x+delta)] = False
            self.remove_range(x, x+delta, dim)

    def remove_indices(self, indices):
        """
        Removes observations of given indices.

        Args:
            ind (list, numpy.ndarray): Array of indexes of the data to remove.
        """
        if isinstance(indices, list):
            indices = np.array(indices)
        elif not isinstance(indices, np.ndarray):
            raise ValueError("indices must be list or numpy array")
        self.mask[indices] = False
    
    ################################################################
    
    def get_prediction_data(self):
        """
        Returns the prediction points.

        Returns:
            numpy.ndarray: X prediction of shape (data_points,input_dims).

        Examples:
            >>> x = data.get_prediction_data()
        """
        if self.X_pred is None:
            return self.X
        return self.X_pred

    def set_prediction_data(self, X):
        """
        Set the prediction points.

        Args:
            X (list,numpy.ndarray): Array of shape (data_points,), (data_points,input_dims), or [(data_points,)] * input_dims used for predictions.

        Examples:
            >>> data.set_prediction_data([5.0, 5.5, 6.0, 6.5, 7.0])
        """
        X_pred, _ = self._format_X(X)
        if X_pred.shape[1] != self.X.shape[1]:
            raise ValueError("X must have the same number of input dimensions as the data")
        self.X_pred = X_pred

    def set_prediction_range(self, start=None, end=None, n=None, step=None):
        """
        Sets the prediction range. The interval is set as `[start,end]`, with either `n` points or a given step between the points.

        Args:
            start (float, str, list): Start of interval, defaults to the first observation.
            end (float, str, list): End of interval, defaults to the last observation.
            n (int, list): Number of points to generate in the interval.
            step (float, str, list): Spacing between points in the interval.

            If neither step or n is passed, default number of points is 100.

        Examples:
            >>> data = mogptk.LoadFunction(lambda x: np.sin(3*x[:,0]), 0, 10, n=200, var=0.1, name='Sine wave')
            >>> data.set_prediction_range(3, 8, 200)
        
            >>> data = mogptk.LoadCSV('gold.csv', 'Date', 'Price')
            >>> data.set_prediction_range('2016-01-15', '2016-06-15', step='1D')
        """
        if start is None:
            start = [np.min(self.X[:,i]) for i in range(self.get_input_dims())]
        if end is None:
            end = [np.max(self.X[:,i]) for i in range(self.get_input_dims())]
        
        start = self._normalize_x_val(start)
        end = self._normalize_x_val(end)
        n = self._normalize_val(n)
        step = self._normalize_val(step)
        for i in range(self.get_input_dims()):
            if n is not None and not isinstance(n[i], int):
                raise ValueError("n must be integer")
            if step is not None and _is_datetime64(self.X_dtypes[i]):
                step[i] = _parse_delta(step[i], self.X_dtypes[i])

        if np.any(end <= start):
            raise ValueError("start must be lower than end")

        # TODO: prediction range for multi input dimension; fix other axes to zero so we can plot?
        X_pred = [np.array([])] * self.get_input_dims()
        for i in range(self.get_input_dims()):
            if n is not None and n[i] is not None:
                X_pred[i] = start[i] + (end[i]-start[i])*np.linspace(0.0, 1.0, n[i])
            else:
                if step is None or step[i] is None:
                    x_step = (end[i]-start[i])/100
                else:
                    x_step = _parse_delta(step[i], self.X_dtypes[i])
                X_pred[i] = np.arange(start[i], end[i]+x_step, x_step)

        n = [X_pred[i].shape[0] for i in range(self.get_input_dims())]
        for i in range(self.get_input_dims()):
            n_tile = np.prod(n[:i])
            n_repeat = np.prod(n[i+1:])
            X_pred[i] = np.tile(np.repeat(X_pred[i], n_repeat), n_tile)
        self.X_pred = np.array(X_pred).T

    ################################################################

    def get_nyquist_estimation(self):
        """
        Estimate the Nyquist frequency by taking 0.5/(minimum distance of points).

        Returns:
            numpy.ndarray: Nyquist frequency array of shape (input_dims,).

        Examples:
            >>> freqs = data.get_nyquist_estimation()
        """
        input_dims = self.get_input_dims()
        nyquist = np.empty((input_dims,))
        for i in range(self.get_input_dims()):
            x = np.sort(self.X[self.mask,i])
            dist = np.abs(x[1:]-x[:-1])
            if len(dist) == 0:
                nyquist[i] = 0.0
            else:
                dist = np.min(dist[np.nonzero(dist)])
                nyquist[i] = 0.5/dist
        return nyquist

    def _get_psd_peaks(self, w, psd):
        # Gaussian: f(x) = A / sqrt(2*pi*C^2) * exp(-(x-B)^2 / (2C^2))
        # i.e. A is the amplitude or peak height, B the mean or peak position, and C the std.dev. or peak width
        peaks, _ = signal.find_peaks(psd)
        if len(peaks) == 0:
            return np.array([]), np.array([]), np.array([])
        peaks = peaks[np.argsort(psd[peaks])[::-1]] # sort by biggest peak first
        peaks = peaks[0.0 < psd[peaks]] # filter out negative peaks which sometimes occur

        widths, _, _, _ = signal.peak_widths(psd, peaks, rel_height=0.5)
        widths *= w[1]-w[0]

        positions = w[peaks]
        variances = widths**2 / (8.0*np.log(2.0)) # from full-width half-maximum to Gaussian sigma
        amplitudes = np.sqrt(psd[peaks])
        return amplitudes, positions, variances

    def get_ls_estimation(self, Q=1, n=10000):
        """
        Peak estimation of the spectrum using Lomb-Scargle.

        Args:
            Q (int): Number of peaks to find.
            n (int): Number of points to use for Lomb-Scargle.

        Returns:
            numpy.ndarray: Amplitude array of shape (Q,input_dims).
            numpy.ndarray: Frequency array of shape (Q,input_dims).
            numpy.ndarray: Variance array of shape (Q,input_dims).

        Examples:
            >>> amplitudes, means, variances = data.get_lombscargle_estimation()
        """
        input_dims = self.get_input_dims()
        A = np.zeros((Q, input_dims))
        B = np.zeros((Q, input_dims))
        C = np.zeros((Q, input_dims))

        nyquist = self.get_nyquist_estimation()
        x, y = self.get_train_data(transformed=True)
        for i in range(input_dims):
            w = np.linspace(0.0, nyquist[i], n)[1:]
            psd = signal.lombscargle(x[:,i]*2.0*np.pi, y, w)
            psd /= x.shape[0]/4.0
            amplitudes, positions, variances = self._get_psd_peaks(w, psd)
            if len(positions) == 0:
                continue
            if Q < len(amplitudes):
                amplitudes = amplitudes[:Q]
                positions = positions[:Q]
                variances = variances[:Q]

            n = len(amplitudes)
            A[:n,i] = amplitudes
            B[:n,i] = positions
            C[:n,i] = variances
        return A, B, C

    def get_bnse_estimation(self, Q=1, n=1000, iters=200):
        """
        Peak estimation of the spectrum using BNSE (Bayesian Non-parametric Spectral Estimation).

        Args:
            Q (int): Number of peaks to find.
            n (int): Number of points of the grid to evaluate frequencies.
            iters (str): Maximum iterations.

        Returns:
            numpy.ndarray: Amplitude array of shape (Q,input_dims).
            numpy.ndarray: Frequency array of shape (Q,input_dims).
            numpy.ndarray: Variance array of shape (Q,input_dims).

        Examples:
            >>> amplitudes, means, variances = data.get_bnse_estimation()
        """
        input_dims = self.get_input_dims()
        A = np.zeros((Q, input_dims))
        B = np.zeros((Q, input_dims))
        C = np.zeros((Q, input_dims))

        nyquist = self.get_nyquist_estimation()
        x, y = self.get_train_data(transformed=True)
        y_err = None
        if self.Y_err is not None:
            y_err_lower = self.Y_transformer.forward(y - self.Y_err[self.mask], x)
            y_err_upper = self.Y_transformer.forward(y + self.Y_err[self.mask], x)
            y_err = (y_err_upper-y_err_lower)/2.0 # TODO: strictly incorrect: takes average error after transformation
        for i in range(input_dims):
            w, psd, _ = BNSE(x[:,i], y, y_err=y_err, max_freq=nyquist[i], n=n, iters=iters)
            # TODO: why? emperically found
            psd /= (np.max(x[:,i])-np.min(x[:,i]))**2
            psd *= np.pi
            amplitudes, positions, variances = self._get_psd_peaks(w, psd)
            if len(positions) == 0:
                continue

            if Q < len(amplitudes):
                amplitudes = amplitudes[:Q]
                positions = positions[:Q]
                variances = variances[:Q]

            num = len(amplitudes)
            A[:num,i] = amplitudes
            B[:num,i] = positions
            C[:num,i] = variances
        return A, B, C

    def get_sm_estimation(self, Q=1, method='LS', optimizer='Adam', iters=200, params={}):
        """
        Peak estimation of the spectrum using the spectral mixture kernel.

        Args:
            Q (int): Number of peaks to find.
            method (str): Method of estimation.
            optimizer (str): Optimization method.
            iters (str): Maximum iterations.
            params (object): Additional parameters for the PyTorch optimizer.

        Returns:
            numpy.ndarray: Amplitude array of shape (Q,input_dims).
            numpy.ndarray: Frequency array of shape (Q,input_dims).
            numpy.ndarray: Variance array of shape (Q,input_dims).

        Examples:
            >>> amplitudes, means, variances = data.get_sm_estimation()
        """
        from .models.sm import SM

        input_dims = self.get_input_dims()
        A = np.zeros((Q, input_dims))
        B = np.zeros((Q, input_dims))
        C = np.zeros((Q, input_dims))

        sm = SM(self, Q)
        sm.init_parameters(method)
        sm.train(method=optimizer, iters=iters, **params)

        for q in range(Q):
            A = sm.gpr.kernel[0].magnitude.numpy().reshape(-1,1).repeat(input_dims, axis=1)
            B = sm.gpr.kernel[0].mean.numpy()
            C = sm.gpr.kernel[0].variance.numpy()
        return A, B, C

    def plot(self, pred=None, title=None, ax=None, legend=True, errorbars=True, transformed=False):
        """
        Plot the data including removed observations, latent function, and predictions.

        Args:
            pred (str): Specify model name to draw.
            title (str): Set the title of the plot.
            ax (matplotlib.axes.Axes): Draw to this axes, otherwise draw to the current axes.
            legend (boolean): Display legend.
            errorbars (boolean): Plot data error bars if available.
            transformed (boolean): Display transformed Y data as used for training.

        Returns:
            matplotlib.axes.Axes

        Examples:
            >>> ax = data.plot()
        """
        # TODO: ability to plot conditional or marginal distribution to reduce input dims
        if self.get_input_dims() > 2:
            raise ValueError("cannot plot more than two input dimensions")
        if self.get_input_dims() == 2:
            raise NotImplementedError("two dimensional input data not yet implemented")

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(12,4), squeeze=True, constrained_layout=True)

        legends = []
        if errorbars and self.Y_err is not None:
            x, y = self.get_train_data(transformed=transformed)
            yl = self.Y[self.mask] - self.Y_err[self.mask]
            yu = self.Y[self.mask] + self.Y_err[self.mask]
            if transformed:
                yl = self.Y_transformer.forward(yl, x)
                yu = self.Y_transformer.forward(yu, x)
            x = x.astype(self.X_dtypes[0])
            ax.errorbar(x, y, [y-yl, yu-y], elinewidth=1.5, ecolor='lightgray', capsize=0, ls='', marker='')

        if self.X_pred is None:
            xmin = np.min(self.X)
            xmax = np.max(self.X)
        else:
            xmin = min(np.min(self.X), np.min(self.X_pred))
            xmax = max(np.max(self.X), np.max(self.X_pred))

        if self.F is not None:
            if _is_datetime64(self.X_dtypes[0]):
                dt = np.timedelta64(1,_get_time_unit(self.X_dtypes[0]))
                n = int((xmax-xmin) / dt) + 1
                x = np.arange(xmin, xmax+np.timedelta64(1,'us'), dt, dtype=self.X_dtypes[0])
            else:
                n = len(self.X[0])*10
                x = np.linspace(xmin, xmax, n)

            y = self.F(x)
            if transformed:
                y = self.Y_transformer.forward(y, x)

            ax.plot(x, y, 'g--', lw=1)
            legends.append(plt.Line2D([0], [0], ls='--', color='g', label='True'))

        if self.has_test_data():
            x, y = self.get_test_data(transformed=transformed)
            x = x.astype(self.X_dtypes[0])
            ax.plot(x, y, 'g.', ms=10)
            legends.append(plt.Line2D([0], [0], ls='', color='g', marker='.', ms=10, label='Latent'))

        x, y = self.get_train_data(transformed=transformed)
        x = x.astype(self.X_dtypes[0])
        ax.plot(x, y, 'r.', ms=10)
        legends.append(plt.Line2D([0], [0], ls='', color='r', marker='.', ms=10, label='Observations'))

        if 0 < len(self.removed_ranges[0]):
            for removed_range in self.removed_ranges[0]:
                x0 = removed_range[0].astype(self.X_dtypes[0])
                x1 = removed_range[1].astype(self.X_dtypes[0])
                y0 = ax.get_ylim()[0]
                y1 = ax.get_ylim()[1]
                ax.add_patch(patches.Rectangle(
                    (x0, y0), x1-x0, y1-y0, fill=True, color='xkcd:strawberry', alpha=0.4, lw=0,
                ))
            legends.insert(0, patches.Rectangle(
                (1, 1), 1, 1, fill=True, color='xkcd:strawberry', alpha=0.4, lw=0, label='Removed Ranges'
            ))

        xmin = xmin.astype(self.X_dtypes[0])
        xmax = xmax.astype(self.X_dtypes[0])
        ax.set_xlim(xmin-(xmax-xmin)*0.001, xmax+(xmax-xmin)*0.001)
        ax.set_xlabel(self.X_labels[0], fontsize=14)
        ax.set_ylabel(self.Y_label, fontsize=14)
        ax.set_title(self.name if title is None else title, fontsize=16)

        if legend:
            ax.legend(handles=legends, ncol=5)
        return ax

    def plot_spectrum(self, title=None, method='ls', ax=None, per=None, maxfreq=None, log=False, transformed=True, n=10000):
        """
        Plot the spectrum of the data. By default it plots up to 99% of the total area under the PSD.

        Args:
            title (str): Set the title of the plot.
            method (str): Set the method to get the spectrum such as LS or BNSE.
            ax (matplotlib.axes.Axes): Draw to this axes, otherwise draw to the current axes.
            per (str, float, numpy.timedelta64): Set the scale of the X axis depending on the formatter used, eg. per=5, per='day', or per='3D'.
            maxfreq (float): Maximum frequency to plot, otherwise the Nyquist frequency is used.
            log (boolean): Show X and Y axis in log-scale.
            transformed (boolean): Display transformed Y data as used for training.
            n (int): Number of points used for periodogram.

        Returns:
            matplotlib.axes.Axes

        Examples:
            >>> ax = data.plot_spectrum(method='bnse')
        """
        # TODO: ability to plot conditional or marginal distribution to reduce input dims
        if self.get_input_dims() > 2:
            raise ValueError("cannot plot more than two input dimensions")
        if self.get_input_dims() == 2:
            raise NotImplementedError("two dimensional input data not yet implemented")

        ax_set = ax is not None
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(12,4), squeeze=True, constrained_layout=True)
        
        X_scale = 1.0
        if _is_datetime64(self.X_dtypes[0]):
            if per is None:
                per = _datetime64_unit_names[_get_time_unit(self.X_dtypes[0])]
            else:
                X_scale = 1.0/_parse_delta(per, self.X_dtypes[0])
                if not isinstance(per, str):
                    per = '%s' % (unit,)

        if per is not None:
            ax.set_xlabel('Frequency [1/'+per+']', fontsize=14)
        else:
            ax.set_xlabel('Frequency', fontsize=14)
        
        X = self.X
        Y = self.Y
        if transformed:
            Y = self.Y_transformer.forward(Y, X)

        idx = np.argsort(X[:,0])
        X = X[idx,0] * X_scale
        Y = Y[idx]

        nyquist = maxfreq
        if nyquist is None:
            dist = np.abs(X[1:]-X[:-1])
            nyquist = float(0.5 / np.average(dist))

        Y_freq_err = np.array([])
        if method.lower() == 'ls':
            X_freq = np.linspace(0.0, nyquist, n+1)[1:]
            Y_freq = signal.lombscargle(X*2.0*np.pi, Y, X_freq)
        elif method.lower() == 'bnse':
            X_freq, Y_freq, Y_freq_err = BNSE(X, Y, max_freq=nyquist, n=n)
        else:
            raise ValueError('periodogram method "%s" does not exist' % (method))

        Y_freq /= Y_freq.sum()*(X_freq[1]-X_freq[0]) # normalize

        if maxfreq is None:
            # cutoff at 99%
            idx = np.cumsum(Y_freq)*(X_freq[1]-X_freq[0]) < 0.99
            X_freq = X_freq[idx]
            Y_freq = Y_freq[idx]
            if len(Y_freq_err) != 0:
                Y_freq_err = Y_freq_err[idx]

        ax.plot(X_freq, Y_freq, '-', c='k', lw=2)
        if len(Y_freq_err) != 0:
            Y_freq_err = 2.0*np.sqrt(Y_freq_err)
            ax.fill_between(X_freq, Y_freq-Y_freq_err, Y_freq+Y_freq_err, color='k', alpha=0.2)
        ax.set_title((self.name + ' Spectrum' if self.name is not None else '') if title is None else title, fontsize=16)

        if log:
            ax.set_xscale('log')
            ax.set_yscale('log')
        else:
            ax.set_ylim(0, None)

        if not ax_set:
            xmin = X_freq.min()
            xmax = X_freq.max()
            ax.set_xlim(xmin-(xmax-xmin)*0.005, xmax+(xmax-xmin)*0.005)
        ax.set_yticks([])
        return ax

    def _normalize_val(self, val):
        # normalize input values, that is: expand to input_dims if a single value
        if val is None:
            return val
        if isinstance(val, np.ndarray):
            if val.ndim == 0:
                val = [val.item()]
            else:
                val = list(val)
        elif _is_iterable(val):
            val = list(val)
        else:
            val = [val] * self.get_input_dims()
        if len(val) != self.get_input_dims():
            raise ValueError("value must be a scalar or a list of values for each input dimension")
        return val

    def _normalize_x_val(self, val):
        # normalize input values for X axis, that is: expand to input_dims if a single value, convert values to appropriate dtype
        val = self._normalize_val(val)
        for i in range(self.get_input_dims()):
            try:
                val[i] = np.array(val[i]).astype(self.X_dtypes[i]).astype(np.float64)
            except:
                raise ValueError("value must be of type %s" % (self.X_dtypes[i],))
        return val

def _is_iterable(val):
    return isinstance(val, collections.abc.Iterable) and not isinstance(val, (dict, str))

def _is_homogeneous_type(seq):
    it = iter(seq)
    first = type(next(it))
    return all(type(x) is first for x in it)

def _check_function(f, input_dims, is_datetime64):
    if not inspect.isfunction(f):
        raise ValueError("must pass a function with input dimensions as parameters")

    sig = inspect.signature(f)
    if len(sig.parameters) != input_dims:
        raise ValueError("must pass a function with input dimensions as parameters")

    x = [np.array([np.datetime64('2000', 'us')]) if is_datetime64[i] else np.ones((1,)) for i in range(input_dims)]

    y = f(*x)
    if y.ndim != 1 or y.shape[0] != 1:
        raise ValueError("function must return Y with shape (data_points,), note that all inputs are of shape (data_points,)")

_datetime64_unit_names = {
    'Y': 'year',
    'M': 'month',
    'W': 'week',
    'D': 'day',
    'h': 'hour',
    'm': 'minute',
    's': 'second',
    'ms': 'millisecond',
    'us': 'microsecond',
}
    
duration_regex = re.compile(
    r'^((?P<years>[\.\d]+?)y)?'
    r'((?P<months>[\.\d]+?)M)?'
    r'((?P<weeks>[\.\d]+?)W)?'
    r'((?P<days>[\.\d]+?)D)?'
    r'((?P<hours>[\.\d]+?)h)?'
    r'((?P<minutes>[\.\d]+?)m)?'
    r'((?P<seconds>[\.\d]+?)s)?$'
    r'((?P<milliseconds>[\.\d]+?)ms)?$'
    r'((?P<microseconds>[\.\d]+?)us)?$'
)

def _parse_delta(text, dtype):
    if np.issubdtype(dtype, np.datetime64):
        dtype = 'timedelta64[%s]' % str(dtype)[-2]

    val = None
    if not isinstance(text, str):
        val = np.array(text)
    elif text == 'year' or text == 'years':
        val = np.timedelta64(1,'Y')
    elif text == 'month' or text == 'months':
        val = np.timedelta64(1,'M')
    elif text == 'week' or text == 'weeks':
        val = np.timedelta64(1,'W')
    elif text == 'day' or text == 'days':
        val = np.timedelta64(1,'D')
    elif text == 'hour' or text == 'hours':
        val = np.timedelta64(1,'h')
    elif text == 'minute' or text == 'minutes':
        val = np.timedelta64(1,'m')
    elif text == 'second' or text == 'seconds':
        val = np.timedelta64(1,'s')
    elif text == 'millisecond' or text == 'milliseconds':
        val = np.timedelta64(1,'ms')
    elif text == 'microsecond' or text == 'microseconds':
        val = np.timedelta64(1,'us')
    if val is not None:
        return val.astype(dtype).astype(np.float64)

    m = duration_regex.match(text)
    if m is None:
        raise ValueError('duration string must be of the form 2h45m, allowed characters: (Y)ear, (M)onth, (W)eek, (D)ay, (h)our, (m)inute, (s)econd, (ms) for milliseconds, (us) for microseconds')

    delta = 0
    matches = m.groupdict()
    if matches['years']:
        delta += np.timedelta64(np.int32(matches['years']),'Y')
    if matches['months']:
        delta += np.timedelta64(np.int32(matches['months']),'M')
    if matches['weeks']:
        delta += np.timedelta64(np.int32(matches['weeks']),'W')
    if matches['days']:
        delta += np.timedelta64(np.int32(matches['days']),'D')
    if matches['hours']:
        delta += np.timedelta64(np.int32(matches['hours']),'h')
    if matches['minutes']:
        delta += np.timedelta64(np.int32(matches['minutes']),'m')
    if matches['seconds']:
        delta += np.timedelta64(np.int32(matches['seconds']),'s')
    if matches['milliseconds']:
        delta += np.timedelta64(np.int32(matches['milliseconds']),'ms')
    if matches['microseconds']:
        delta += np.timedelta64(np.int32(matches['microseconds']),'us')
    return delta.astype(dtype).astype(np.float64)

def _datetime64_to_higher_unit(array):
    if array.dtype in ['<M8[Y]', '<M8[M]', '<M8[W]', '<M8[D]']:
        return array

    units = ['D', 'h', 'm', 's']  # cannot convert days to non-linear months or years
    for unit in units:
        frac, _ = np.modf((array-np.datetime64('2000')) / np.timedelta64(1,unit))
        if not np.any(frac):
            return array.astype('datetime64[%s]' % (unit,))
    return array

def _timedelta64_to_higher_unit(array):
    if array.dtype in ['<m8[Y]', '<m8[M]', '<m8[W]', '<m8[D]']:
        return array

    units = ['D', 'h', 'm', 's']  # cannot convert days to non-linear months or years
    for unit in units:
        frac, _ = np.modf(array / np.timedelta64(1,unit))
        if not np.any(frac):
            return array.astype('timedelta64[%s]' % (unit,))
    return array

def _is_datetime64(dtype):
    return np.issubdtype(dtype, np.datetime64)

def _get_time_unit(dtype):
    unit = str(dtype)
    locBracket = unit.find('[')
    if locBracket == -1:
        return ''
    return unit[locBracket+1:-1]
