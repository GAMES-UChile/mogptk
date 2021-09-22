import re
import copy
import inspect
import datetime
import logging
import math
import collections

import numpy as np
import pandas as pd
from scipy import signal
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pandas.plotting import register_matplotlib_converters

from .bnse import bse
from .serie import Serie, TransformLinear
from .plot import plot_spectrum

register_matplotlib_converters()

logger = logging.getLogger('mogptk')

def LoadFunction(f, start, end, n, var=0.0, name="", random=False):
    """
    LoadFunction loads a dataset from a given function y = f(x) + Normal(0,var). It will pick `n` data points between start and end for the X axis for which `f` is being evaluated. By default the `n` points are spread equally over the interval, with `random=True` they will be picked randomly.

    The given function should take one argument X which is a list of `numpy.ndarray` of shape (n,) for every input dimension and returns an `numpy.ndarray` Y of shape (n,). If your data has only one input dimension, you can use X[0] to select the first (and only) input dimension.

    Args:
        f (function): Function taking X a list with elements of shape (n,) for each input dimension and returning shape (n,) as Y.
        start (float, list): Define start of interval.
        end (float, list): Define end of interval.
        n (int, list): Number of data points to pick between start and end.
        var (float): Variance added to the output.
        name (str): Name of data.
        random (boolean): Select points randomly between start and end.

    Returns:
        mogptk.data.Data

    Examples:
        >>> LoadFunction(lambda x: np.sin(3*x[:,0]), 0, 10, n=200, var=0.1, name='Sine wave')
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

        N_tile = math.prod(n[:i])
        N_repeat = math.prod(n[i+1:])
        x[i] = np.tile(np.repeat(x[i], N_repeat), N_tile)

    y = f(x)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:,0]
    N = math.prod(n)
    y += np.random.normal(0.0, var, (N,))

    data = Data(x, y, name=name)
    data.set_function(f)
    return data

################################################################
################################################################
################################################################

class Data:
    def __init__(self, X, Y, name=None, x_labels=None, y_label=None):
        """
        Data class that holds all observations, latent functions and predicted data.

        This class accepts the data directly, otherwise you can load data conveniently using `LoadFunction`, `LoadCSV`, `LoadDataFrame`, etc. The data class allows to modify the data before passing into the model. Examples are transforming data, such as detrending or taking the log, removing data ranges to simulate sensor failure, and aggregating data for given spans on X, such as aggregating daily data into weekly data. Additionally, we also use this class to set the range we want to predict.

        It is possible to use the format given by `numpy.meshgrid` for X as a list of numpy arrays for each input dimension, and its values in Y. Each input dimension and Y must have shape (N1,N2,...,Nn) where n is the number of input dimensions and N the number of data points per input dimension.

        Args:
            X (list, numpy.ndarray, dict): Independent variable data of shape (n,) or (n,input_dims), or a list with elements of shape (n,) for each input dimension.
            Y (list, numpy.ndarray): Dependent variable data of shape (n,).
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
                    raise ValueError("X dict should contain all lists or np.ndarrays where each has the same length")
                if not all(key in X for key in x_labels):
                    raise ValueError("X dict must contain all keys listed in x_labels")
                X = [X[key] for key in x_labels]

        # check if X is correct
        if isinstance(X, list):
            if all(isinstance(x, list) for x in X):
                m = len(X[0])
                if not all(len(x) == m for x in X[1:]):
                    raise ValueError("X list items must all be lists of the same length")
                if not all(all(isinstance(val, (int, float, datetime.datetime, np.datetime64)) for val in x) for x in X):
                    raise ValueError("X list items must all be lists of numbers or datetime")
                if not all(_is_homogeneous_type(x) for x in X):
                    raise ValueError("X list items must all be lists with elements of the same type")
            elif all(isinstance(x, np.ndarray) for x in X):
                m = len(X[0])
                if not all(len(x) == m for x in X[1:]):
                    raise ValueError("X list items must all be numpy.ndarrays of the same length")
            elif not all(isinstance(x, (int, float, datetime.datetime, np.datetime64)) for x in X):
                raise ValueError("X list items must be all lists, all numpy.ndarrays, or all numbers or datetime")
            elif not _is_homogeneous_type(X):
                raise ValueError("X list items must all have elements of the same type")
            X = [np.array(x) for x in X]
        elif isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            if X.ndim != 2:
                raise ValueError("X must be either a one or two dimensional array of data")
            X = [X[:,i] for i in range(X.shape[1])]
        else:
            raise ValueError("X must be list or numpy array, if dict is passed then x_labels must also be set")

        input_dims = len(X)
        # try to cast unknown data types, X becomes np.float64 or np.datetime64
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

        # check if Y is correct
        if isinstance(Y, list):
            if not all(isinstance(y, (int, float)) for y in Y):
                raise ValueError("Y list items must all be numbers")
            elif not _is_homogeneous_type(Y):
                raise ValueError("Y list items must all have elements of the same type")
            Y = np.array(Y)
        elif not isinstance(Y, np.ndarray):
            raise ValueError("Y must be list or numpy array")

        # try to cast unknown data types, Y becomes np.float64
        try:
            Y = Y.astype(np.float64)
        except:
            raise ValueError("Y data must have a number data type")

        # convert meshgrids to flat arrays
        if 1 < X[0].ndim and 1 < Y.ndim and X[0].shape == Y.shape:
            X = [np.ravel(x) for x in X]
            Y = np.ravel(Y)

        if any(x.ndim != 1 for x in X):
            raise ValueError("X must be a one dimensional array of data for every input dimension")
        if Y.ndim != 1:
            raise ValueError("Y must be a one dimensional array of data")
        if Y.shape[0] == 0:
            raise ValueError("X and Y must have a length greater than zero")
        if any(x.shape[0] != Y.shape[0] for x in X):
            raise ValueError("X and Y must be of the same length for each input dimension")


        self.X = [Serie(X[i]) for i in range(input_dims)] # [shape (n)] * input_dims
        self.Y = Serie(Y) # shape (n)
        self.mask = np.array([True] * Y.shape[0])
        self.F = None
        self.X_pred = self.X
        self.Y_mu_pred = {}
        self.Y_var_pred = {}
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

    def __repr__(self):
        df = pd.DataFrame()
        for i in range(len(self.X)):
            df[self.X_labels[i]] = self.X[i]
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
    
        The function should take one argument X of shape (n,input_dims) and return Y of shape (n,). If your data has only one input dimension, you can use X[:,0] to select the first (and only) input dimension.

        Args:
            f (function): Function taking X with shape (n,input_dims) and returning shape (n,) as Y.

        Examples:
            >>> data.set_function(lambda x: np.sin(3*x[:,0])
        """
        _check_function(f, self.get_input_dims(), [x.is_datetime64() for x in self.X])
        self.F = f

    def rescale_x(self, upper=1000.0):
        """
        Rescale the X axis so that it is in the interval [0.0,upper]. This helps training most kernels.

        Args:
            upper (float): Upper end of the interval.

        Examples:
            >>> data.rescale_x()
        """

        for i in range(self.get_input_dims()):
            X = self.X[i].transformed
            xmin = np.min(X)
            xmax = np.max(X)
            t = TransformLinear(xmin, (xmax-xmin)/upper)
            t.set_data(X)
            self.X[i].apply(t)

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

        t = transformer
        if isinstance(t, type):
            t = transformer()
        else:
            t = copy.deepcopy(t)
        t.set_data(self)

        self.Y.apply(t, np.array([x for x in self.X]).T)
    
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
            ind = np.logical_and(self.X[dim] >= start[dim], self.X[dim] < end[dim])
        else:
            ind = np.logical_and(self.X[0] >= start[0], self.X[0] < end[0])
            for i in range(1,self.get_input_dims()):
                ind = np.logical_and(ind, np.logical_and(self.X[i] >= start[i], self.X[i] < end[i]))

        self.X = [x[ind] for x in self.X]
        self.Y = self.Y[ind]
        self.mask = self.mask[ind]

    def aggregate(self, duration, f=np.mean, dim=0):
        """
        Aggregate the data by duration and apply a function to obtain a reduced dataset.

        For example, group daily data by week and take the mean. The duration can be set as a number which defined the intervals on the X axis, or by a string written in the duration format in case the X axis has data type `numpy.datetime64`. The duration format uses: Y=year, M=month, W=week, D=day, h=hour, m=minute, and s=second. For example, 3W1D means three weeks and one day, ie. 22 days, or 6M to mean six months.

        Args:
            duration (float, str): Duration along the X axis or as a string in the duration format.
            f (function): Function to use to reduce data mapping a numpy array to a scalar, such as `numpy.mean`.
            dim (int): Input dimension to apply to, defaults to the first input dimension.

        Examples:
            >>> data.aggregate(5)

            >>> data.aggregate('2W', f=np.sum)
        """
        start = np.min(self.X[dim])
        end = np.max(self.X[dim])
        step = _parse_delta(duration)

        X = np.arange(start+step/2, end+step/2, step)
        Y = np.empty((len(X)))
        for i in range(len(X)):
            ind = (self.X[dim] >= X[i]-step/2) & (self.X[dim] < X[i]+step/2)
            Y[i] = f(self.Y[ind])

        self.X[dim] = Serie(X, self.X[dim].transformers)
        self.Y = Serie(Y, self.Y.transformers, self.X)
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
        return len(self.X)
    
    def get_data(self, transformed=False):
        """
        Returns all observations, train and test.

        Arguments:
            transformed (boolean): Return transformed data.

        Returns:
            list of numpy.ndarray: X data of shape [(n,)] * input_dims.
            numpy.ndarray: Y data of shape (n,).

        Examples:
            >>> x, y = data.get_data()
        """
        if transformed:
            return self.X, self.Y.transformed
        return self.X, np.array(self.Y)

    def get_train_data(self, transformed=False):
        """
        Returns the observations used for training.

        Arguments:
            transformed (boolean): Return transformed data.

        Returns:
            list of numpy.ndarray: X data of shape [(n,)] * input_dims.
            numpy.ndarray: Y data of shape (n,).

        Examples:
            >>> x, y = data.get_train_data()
        """
        if transformed:
            return [x[self.mask] for x in self.X], self.Y.transformed[self.mask]
        return [x[self.mask] for x in self.X], np.array(self.Y[self.mask])

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
        X = [x[~self.mask] for x in self.X]
        if self.F is not None:
            if X.shape[0] == 0:
                X, _ = self.get_data()
            Y = self.F(X)
            if transformed:
                Y = self.Y.transform(Y, X)
            return X, Y
        if transformed:
            return X, self.Y.transformed[~self.mask]
        return X, np.array(self.Y[~self.mask])

    ################################################################

    def reset(self):
        """
        Reset the data set and undo the removal of data points. That is, this reverts any calls to `remove_randomly`, `remove_range`, `remove_relative_range`, `remove_random_ranges`, and `remove_index`.
        """
        self.mask[:] = True
        for i in range(len(self.removed_ranges)):
            self.removed_ranges[i] = []
    
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

        idx = np.random.choice(len(self.Y), n, replace=False)
        self.mask[idx] = False
    
    def remove_range(self, start=None, end=None, dim=None):
        """
        Removes observations in the interval `[start,end]`.
        
        Args:
            start (float, str): Start of interval. Defaults to the first value in observations.
            end (float, str): End of interval (not included). Defaults to the last value in observations.
            dim (int): Input dimension to apply to, if not specified applies to all input dimensions.

        Examples:
            >>> data = mogptk.LoadFunction(lambda x: np.sin(3*x[:,0]), 0, 10, n=200, var=0.1, name='Sine wave')
            >>> data.remove_range(3, 8)
        
            >>> data = mogptk.LoadCSV('gold.csv', 'Date', 'Price')
            >>> data.remove_range('2016-01-15', '2016-06-15')
        """
        if start is None:
            start = [np.min(x) for x in self.X]
        if end is None:
            end = [np.max(x) for x in self.X]

        start = self._normalize_x_val(start)
        end = self._normalize_x_val(end)

        if dim is not None:
            mask = np.logical_and(self.X[dim] >= start[dim], self.X[dim] < end[dim])
            self.removed_ranges[dim].append([start[dim], end[dim]])
        else:
            mask = np.logical_and(self.X[0] >= start[0], self.X[0] < end[0])
            for i in range(1,self.get_input_dims()):
                mask = np.logical_or(mask, np.logical_and(self.X[i] >= start[i], self.X[i] < end[i]))
            for i in range(self.get_input_dims()):
                self.removed_ranges[i].append([start[i], end[i]])
        self.mask[np.where(mask)] = False
    
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

        x_min = [np.min(x) for x in self.X]
        x_max = [np.max(x) for x in self.X]
        for i in range(self.get_input_dims()):
            start[i] = x_min[i] + max(0.0, min(1.0, start[i])) * (x_max[i]-x_min[i])
            end[i] = x_min[i] + max(0.0, min(1.0, end[i])) * (x_max[i]-x_min[i])

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

        delta = _parse_delta(duration)
        m = (np.max(self.X[dim])-np.min(self.X[dim])) - n*delta
        if m <= 0:
            raise ValueError("no data left after removing ranges")

        locs = self.X[dim] <= (np.max(self.X[dim])-delta)
        locs[sum(locs)] = True # make sure the last data point can be deleted
        for i in range(n):
            if len(self.X[dim][locs]) == 0:
                break # range could not be removed, there is no remaining data range of width delta
            x = self.X[dim][locs][np.random.randint(len(self.X[dim][locs]))]
            locs[(self.X[dim] > x-delta) & (self.X[dim] < x+delta)] = False
            self.mask[(self.X[dim] >= x) & (self.X[dim] < x+delta)] = False
            self.removed_ranges[dim].append([x, x+delta])

    def remove_index(self, index):
        """
        Removes observations of given index

        Args:
            index(list, numpy.ndarray): Array of indexes of the data to remove.
        """
        if isinstance(index, list):
            index = np.array(index)
        elif not isinstance(index, np.ndarray):
            raise ValueError("index must be list or numpy array")

        self.mask[index] = False
    
    ################################################################
    
    def get_prediction_names(self):
        """
        Returns the model names of the saved predictions.

        Returns:
            list: List of prediction names.

        Examples:
            >>> data.get_prediction_names()
            ['MOSM', 'CSM', 'SM-LMC', 'CONV']
        """
        return self.Y_mu_pred.keys()
    
    def get_prediction_x(self):
        """
        Returns the prediction X range.

        Returns:
            numpy.ndarray: X prediction of shape [(n,)] * input_dims.

        Examples:
            >>> x = data.get_prediction_x()
        """
        return self.X_pred.copy()
    
    def get_prediction(self, name, sigma=2.0, transformed=False):
        """
        Returns the prediction of a given name with a confidence interval of `sigma` times the standard deviation.

        Args:
            name (str): Name of the model of the prediction.
            sigma (float): The confidence interval's number of standard deviations.
            transformed (boolean): Return transformed data as used for training.

        Returns:
            numpy.ndarray: X prediction of shape [(n,)] * input_dims.
            numpy.ndarray: Y mean prediction of shape (n,).
            numpy.ndarray: Y lower prediction of uncertainty interval of shape (n,).
            numpy.ndarray: Y upper prediction of uncertainty interval of shape (n,).

        Examples:
            >>> x, y_mean, y_var_lower, y_var_upper = data.get_prediction('MOSM', sigma=1)
        """
        if name not in self.Y_mu_pred:
            raise ValueError("prediction name '%s' does not exist" % (name))
       
        X = self.X_pred.copy()
        mu = self.Y_mu_pred[name]
        lower = mu - sigma * np.sqrt(self.Y_var_pred[name])
        upper = mu + sigma * np.sqrt(self.Y_var_pred[name])

        if transformed:
            return X, mu, lower, upper

        mu = Serie(self.Y.detransform(mu, X), self.Y.transformers, transformed=mu)
        lower = Serie(self.Y.detransform(lower, X), self.Y.transformers, transformed=lower)
        upper = Serie(self.Y.detransform(upper, X), self.Y.transformers, transformed=upper)
        return X, mu, lower, upper
    
    def set_prediction_x(self, X):
        """
        Set the prediction range directly for saved predictions. This will clear old predictions.

        Args:
            X (list, numpy.ndarray): Array of shape (n,), (n,input_dims), or [(n,)] * input_dims used for predictions.

        Examples:
            >>> data.set_prediction_x([5.0, 5.5, 6.0, 6.5, 7.0])
        """
        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(-1,1)
            if X.ndim != 2 or X.shape[1] != self.get_input_dims():
                raise ValueError("X shape must be (n,input_dims)")
            X = [X[:,i] for i in range(self.get_input_dims())]
        elif not isinstance(X, list):
            raise ValueError("X expected to be a list or numpy.ndarray")
        if not all(x.ndim == 1 for x in X) or len(X) != self.get_input_dims():
            raise ValueError("X shape must be (n,), (n,input_dims), or [(n,)] * input_dims")

        X = [X[i].astype(self.X[i].dtype) for i in range(self.get_input_dims())]
        self.X_pred = [Serie(X[i], self.X[i].transformers) for i in range(self.get_input_dims())]

        # clear old prediction data now that X_pred has been updated
        self.clear_predictions()

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
            start = [x[0] for x in self.X]
        if end is None:
            start = [x[-1] for x in self.X]
        
        start = self._normalize_x_val(start)
        end = self._normalize_x_val(end)
        n = self._normalize_val(n)
        step = self._normalize_val(step)
        for i in range(self.get_input_dims()):
            if n is not None and not isinstance(n[i], int):
                raise ValueError("n must be integer")
            if step is not None and np.issubdtype(self.X[i].dtype, np.datetime64):
                step[i] = _parse_delta(step[i])

        if np.any(end <= start):
            raise ValueError("start must be lower than end")

        # TODO: prediction range for multi input dimension; fix other axes to zero so we can plot?
        X_pred = [np.array([])] * self.get_input_dims()
        for i in range(self.get_input_dims()):
            if n is not None and n[i] is not None:
                X_pred[i] = start[i] + (end[i]-start[i])*np.linspace(0.0, 1.0, int(n[i]))
            else:
                if step is None or step[i] is None:
                    x_step = (end[i]-start[i])/100
                else:
                    x_step = _parse_delta(step[i])
                X_pred[i] = np.arange(start[i], end[i]+x_step, x_step)
        self.X_pred = [Serie(x, self.X[i].transformers) for i, x in enumerate(X_pred)]

        # clear old prediction data now that X_pred has been updated
        self.clear_predictions()

    def clear_predictions(self):
        """
        Clear all saved predictions.
        """
        self.Y_mu_pred = {}
        self.Y_var_pred = {}

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
            x = np.sort(self.X[i].transformed[self.mask])
            dist = np.abs(x[1:]-x[:-1])
            dist = np.min(dist[np.nonzero(dist)])
            nyquist[i] = 0.5/dist
        return nyquist

    def get_lombscargle_estimation(self, Q=1, n=10000):
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

        # Gaussian: f(x) = A * exp((x-B)^2 / (2C^2))
        # i.e. A is the amplitude or peak height, B the mean or peak position, and C the std.dev. or peak width
        A = np.zeros((Q, input_dims))
        B = np.zeros((Q, input_dims))
        C = np.zeros((Q, input_dims))

        nyquist = self.get_nyquist_estimation()
        for i in range(input_dims):
            x, y = np.array([x.transformed[self.mask] for x in self.X]).T, self.Y.transformed[self.mask]
            freq = np.linspace(0.0, nyquist[i], n+1)[1:]
            psd = signal.lombscargle(x[:,i]*2.0*np.pi, y, freq)

            ind, _ = signal.find_peaks(psd)
            ind = ind[np.argsort(psd[ind])[::-1]]  # sort by biggest peak first

            widths, width_heights, _, _ = signal.peak_widths(psd, ind, rel_height=0.5)
            widths *= freq[1]-freq[0]

            positions = freq[ind]
            amplitudes = psd[ind]
            # from full-width half-maximum to Gaussian sigma
            # note that amplitudes / width_heights is near 2 when the base of the peak is near zero
            stddevs = widths / np.sqrt(8 * np.log(amplitudes / width_heights)) 

            if Q < len(amplitudes):
                amplitudes = amplitudes[:Q]
                positions = positions[:Q]
                stddevs = stddevs[:Q]

            n = len(amplitudes)
            A[:n,i] = np.sqrt(amplitudes)
            B[:n,i] = positions
            C[:n,i] = stddevs
        return A, B, C

    def get_bnse_estimation(self, Q=1, n=1000):
        """
        Peak estimation of the spectrum using BNSE (Bayesian Non-parametric Spectral Estimation).

        Args:
            Q (int): Number of peaks to find.
            n (int): Number of points of the grid to evaluate frequencies.

        Returns:
            numpy.ndarray: Amplitude array of shape (Q,input_dims).
            numpy.ndarray: Frequency array of shape (Q,input_dims).
            numpy.ndarray: Variance array of shape (Q,input_dims).

        Examples:
            >>> amplitudes, means, variances = data.get_bnse_estimation()
        """
        input_dims = self.get_input_dims()

        # Gaussian: f(x) = A * exp((x-B)^2 / (2C^2))
        # Ie. A is the amplitude or peak height, B the mean or peak position, and C the variance or peak width
        A = np.zeros((Q, input_dims))
        B = np.zeros((Q, input_dims))
        C = np.zeros((Q, input_dims))

        nyquist = self.get_nyquist_estimation()
        for i in range(input_dims):
            x, y = np.array([x.transformed[self.mask] for x in self.X]).T, self.Y.transformed[self.mask]
            bnse = bse(x[:,i], y)
            bnse.set_freqspace(nyquist[i], dimension=n)
            bnse.train()
            bnse.compute_moments()

            amplitudes, positions, variances = bnse.get_freq_peaks()
            if len(positions) == 0:
                continue

            if Q < len(amplitudes):
                amplitudes = amplitudes[:Q]
                positions = positions[:Q]
                variances = variances[:Q]

            num = len(amplitudes)
            # division by 100 makes it similar to other estimators (emperically found)
            A[:num,i] = np.sqrt(amplitudes) / 100
            B[:num,i] = positions
            C[:num,i] = variances
        return A, B, C

    def get_sm_estimation(self, Q=1, method='BNSE', optimizer='Adam', iters=100, params={}, plot=False):
        """
        Peak estimation of the spectrum using the spectral mixture kernel.

        Args:
            Q (int): Number of peaks to find.
            method (str): Method of estimating SM kernels.
            optimizer (str): Optimization method for SM kernels.
            iters (str): Maximum iteration for SM kernels.
            params (object): Additional parameters for the PyTorch optimizer.
            plot (bool): Show the PSD of the kernel after fitting.

        Returns:
            numpy.ndarray: Amplitude array of shape (Q,input_dims).
            numpy.ndarray: Frequency array of shape (Q,input_dims).
            numpy.ndarray: Variance array of shape (Q,input_dims).

        Examples:
            >>> amplitudes, means, variances = data.get_sm_estimation()
        """
        from .models.sm import SM

        input_dims = self.get_input_dims()

        # Gaussian: f(x) = A * exp((x-B)^2 / (2C^2))
        # Ie. A is the amplitude or peak height, B the mean or peak position, and C the variance or peak width
        A = np.zeros((Q, input_dims))
        B = np.zeros((Q, input_dims))
        C = np.zeros((Q, input_dims))

        sm = SM(self, Q)
        sm.init_parameters(method)
        sm.train(method=optimizer, iters=iters, **params)

        if plot:
            nyquist = self.get_nyquist_estimation()
            means = np.array([sm.model.kernel[0][q].mean.numpy() for q in range(Q)])
            weights = np.array([sm.model.kernel[0][q].weight.numpy() for q in range(Q)])
            scales = np.array([sm.model.kernel[0][q].variance.numpy() for q in range(Q)])
            nyquist = np.expand_dims(nyquist, 0)
            means = np.expand_dims(means, 1)
            scales = np.expand_dims(scales, 1)
            plot_spectrum(means, scales, weights=weights, nyquist=nyquist, title=self.name)

        for q in range(Q):
            A[q,:] = sm.kernel[0][q].weight.numpy()  # TODO: weight is not per input_dims
            B[q,:] = sm.kernel[0][q].mean.numpy()
            C[q,:] = sm.kernel[0][q].variance.numpy()
        return A, B, C

    def plot(self, pred=None, title=None, ax=None, legend=True, transformed=False):
        """
        Plot the data including removed observations, latent function, and predictions.

        Args:
            pred (str): Specify model name to draw.
            title (str): Set the title of the plot.
            ax (matplotlib.axes.Axes): Draw to this axes, otherwise draw to the current axes.
            legend (boolean): Display legend.
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
            raise NotImplementedError("two dimensional input data not yet implemented") # TODO

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(12, 3.0), squeeze=True, constrained_layout=True)

        legends = []
        colors = list(matplotlib.colors.TABLEAU_COLORS)
        for i, name in enumerate(self.Y_mu_pred):
            if self.Y_mu_pred[name].size != 0 and (pred is None or name.lower() == pred.lower()):
                X_pred, mu, lower, upper = self.get_prediction(name, transformed=transformed)

                idx = np.argsort(X_pred[0])
                ax.plot(X_pred[0][idx], mu[idx], ls='-', color=colors[i], lw=2)
                ax.fill_between(X_pred[0][idx], lower[idx], upper[idx], color=colors[i], alpha=0.1)
                #ax.plot(X_pred[:,0][idx], lower[idx], ls='-', color=colors[i], lw=1, alpha=0.5)
                #ax.plot(X_pred[:,0][idx], upper[idx], ls='-', color=colors[i], lw=1, alpha=0.5)

                label = 'Prediction'
                if name is not None:
                    label += ' ' + name
                legends.append(plt.Line2D([0], [0], ls='-', color=colors[i], lw=2, label=label))

        if self.F is not None:
            xmin = min(np.min(self.X[0]), np.min(self.X_pred[0]))
            xmax = max(np.max(self.X[0]), np.max(self.X_pred[0]))

            if np.issubdtype(self.X[0].dtype, np.datetime64):
                dt = np.timedelta64(1,self.X[0].get_time_unit())
                n = int((xmax-xmin) / dt) + 1
                x = np.arange(xmin, xmax+np.timedelta64(1,'us'), dt, dtype=self.X[0].dtype)
            else:
                n = len(self.X[0])*10
                x = np.linspace(xmin, xmax, n)

            x = [x]
            y = self.F(x)
            if transformed:
                y = self.Y.transform(y, x)

            ax.plot(x[:,0], y, 'r--', lw=1)
            legends.append(plt.Line2D([0], [0], ls='--', color='r', label='True'))

        _, Y = self.get_data(transformed=transformed)
        idx = np.argsort(self.X[0])
        ax.plot(self.X[0][idx], Y[idx], 'k--', alpha=0.8)
        legends.append(plt.Line2D([0], [0], ls='--', color='k', label='All Points'))

        x, y = self.get_train_data(transformed=transformed)
        if 1000 < x[0].shape[0]:
            ax.plot(x[0], y, 'k-')
            legends.append(plt.Line2D([0], [0], ls='-', color='k', label='Training Points'))
        else:
            ax.plot(x[0], y, 'k.', mew=1, ms=13, markeredgecolor='white')
            legends.append(plt.Line2D([0], [0], ls='', color='k', marker='.', ms=10, label='Training Points'))

        if self.has_test_data():
            for removed_range in self.removed_ranges[0]:
                x0 = removed_range[0]
                x1 = removed_range[1]
                y0 = ax.get_ylim()[0]
                y1 = ax.get_ylim()[1]
                ax.add_patch(patches.Rectangle(
                    (x0, y0), x1-x0, y1-y0, fill=True, color='xkcd:strawberry', alpha=0.2, lw=0,
                ))
            legends.append(patches.Rectangle(
                (1, 1), 1, 1, fill=True, color='xkcd:strawberry', alpha=0.5, lw=0, label='Removed Ranges'
            ))

        xmin = min(np.min(self.X[0]), np.min(self.X_pred[0]))
        xmax = max(np.max(self.X[0]), np.max(self.X_pred[0]))
        ax.set_xlim(xmin - (xmax - xmin)*0.001, xmax + (xmax - xmin)*0.001)

        ax.set_xlabel(self.X_labels[0])
        ax.set_ylabel(self.Y_label)
        ax.set_title(self.name if title is None else title, fontsize=14)

        if legend:
            legend_rows = (len(legends)-1)/5 + 1
            ax.legend(handles=legends, loc="upper center", bbox_to_anchor=(0.5,(3.0+0.5+0.3*legend_rows)/3.0), ncol=5)

        return ax

    def plot_spectrum(self, title=None, method='ls', ax=None, per=None, maxfreq=None, transformed=True):
        """
        Plot the spectrum of the data.

        Args:
            title (str): Set the title of the plot.
            method (str): Set the method to get the spectrum such as LS or BNSE.
            ax (matplotlib.axes.Axes): Draw to this axes, otherwise draw to the current axes.
            per (str, float, numpy.timedelta64): Set the scale of the X axis depending on the formatter used, eg. per=5, per='day', or per='3D'.
            maxfreq (float): Maximum frequency to plot, otherwise the Nyquist frequency is used.
            transformed (boolean): Display transformed Y data as used for training.

        Returns:
            matplotlib.axes.Axes

        Examples:
            >>> ax = data.plot_spectrum(method='bnse')
        """
        # TODO: ability to plot conditional or marginal distribution to reduce input dims
        if self.get_input_dims() > 2:
            raise ValueError("cannot plot more than two input dimensions")
        if self.get_input_dims() == 2:
            raise NotImplementedError("two dimensional input data not yet implemented") # TODO

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(12, 3.0), squeeze=True, constrained_layout=True)
        
        X_scale = 1.0
        if np.issubdtype(self.X[0].dtype, np.datetime64):
            if per is None:
                per = _datetime64_unit_names[self.X[0].get_time_unit()]
            else:
                unit = _parse_delta(per)
                X_scale = np.timedelta64(1,self.X[0].get_time_unit()) / unit
                if not isinstance(per, str):
                    per = '%s' % (unit,)

        if per is not None:
            ax.set_xlabel('Frequency [1/'+per+']')
        else:
            ax.set_xlabel('Frequency')
        
        X = self.X[0].astype(np.float)
        Y = self.Y
        if transformed:
            Y = self.Y.transformed

        idx = np.argsort(X)
        X = X[idx] * X_scale
        Y = Y[idx]

        nyquist = maxfreq
        if nyquist is None:
            dist = np.abs(X[1:]-X[:-1])
            nyquist = 0.5 / np.average(dist)

        X_freq = np.linspace(0.0, nyquist, 10001)[1:]
        Y_freq_err = []
        if method.lower() == 'ls':
            Y_freq = signal.lombscargle(X*2.0*np.pi, Y, X_freq)
        elif method.lower() == 'bnse':
            bnse = bse(X, Y)
            bnse.set_freqspace(nyquist, 10001)
            bnse.train()
            bnse.compute_moments()

            Y_freq = bnse.post_mean_r**2 + bnse.post_mean_i**2
            Y_freq_err = 2 * np.sqrt(np.diag(bnse.post_cov_r**2 + bnse.post_cov_i**2))
            Y_freq = Y_freq[1:]
            Y_freq_err = Y_freq_err[1:]
        else:
            raise ValueError('periodogram method "%s" does not exist' % (method))

        ax.plot(X_freq, Y_freq, '-', c='k', lw=2)
        if len(Y_freq_err) != 0:
            ax.fill_between(X_freq, Y_freq-Y_freq_err, Y_freq+Y_freq_err, alpha=0.4)
        ax.set_title((self.name + ' Spectrum' if self.name is not None else '') if title is None else title, fontsize=14)

        xmin = X_freq.min()
        xmax = X_freq.max()
        ax.set_xlim(xmin - (xmax - xmin)*0.005, xmax + (xmax - xmin)*0.005)
        ax.set_yticks([])
        ax.set_ylim(0, None)
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
                val[i] = self.X[i].dtype.type(val[i])
            except:
                raise ValueError("value must be of type %s" % (self.X[i].dtype,))
        return val

def _is_iterable(val):
    return isinstance(val, collections.abc.Iterable) and not isinstance(val, (dict, str))

def _is_homogeneous_type(seq):
    it = iter(seq)
    first = type(next(it))
    return all(type(x) is first for x in it)

def _check_function(f, input_dims, is_datetime64):
    if not inspect.isfunction(f):
        raise ValueError("function must take X as a parameter")

    sig = inspect.signature(f)
    if not len(sig.parameters) == 1:
        raise ValueError("function must take X as a parameter")

    x = [np.array([np.datetime64('2000', 'us')]) if is_datetime64[i] else np.ones((1,)) for i in range(input_dims)]
    y = f(x)
    if len(y.shape) != 1 or y.shape[0] != 1:
        raise ValueError("function must return Y with shape (n,), note that X is a list where each element is an input dimension of shape (n,)")

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

def _parse_delta(text):
    if not isinstance(text, str):
        return text

    if text == 'year' or text == 'years':
        return np.timedelta64(1, 'Y')
    elif text == 'month' or text == 'months':
        return np.timedelta64(1, 'M')
    elif text == 'week' or text == 'weeks':
        return np.timedelta64(1, 'W')
    elif text == 'day' or text == 'days':
        return np.timedelta64(1, 'D')
    elif text == 'hour' or text == 'hours':
        return np.timedelta64(1, 'h')
    elif text == 'minute' or text == 'minutes':
        return np.timedelta64(1, 'm')
    elif text == 'second' or text == 'seconds':
        return np.timedelta64(1, 's')
    elif text == 'millisecond' or text == 'milliseconds':
        return np.timedelta64(1, 'ms')
    elif text == 'microsecond' or text == 'microseconds':
        return np.timedelta64(1, 'us')

    m = duration_regex.match(text)
    if m is None:
        raise ValueError('duration string must be of the form 2h45m, allowed characters: (Y)ear, (M)onth, (W)eek, (D)ay, (h)our, (m)inute, (s)econd, (ms) for milliseconds, (us) for microseconds')

    delta = 0
    matches = m.groupdict()
    if matches['years']:
        delta += np.timedelta64(np.int32(matches['years']), 'Y')
    if matches['months']:
        delta += np.timedelta64(np.int32(matches['months']), 'M')
    if matches['weeks']:
        delta += np.timedelta64(np.int32(matches['weeks']), 'W')
    if matches['days']:
        delta += np.timedelta64(np.int32(matches['days']), 'D')
    if matches['hours']:
        delta += np.timedelta64(np.int32(matches['hours']), 'h')
    if matches['minutes']:
        delta += np.timedelta64(np.int32(matches['minutes']), 'm')
    if matches['seconds']:
        delta += np.timedelta64(np.int32(matches['seconds']), 's')
    if matches['milliseconds']:
        delta += np.timedelta64(np.int32(matches['milliseconds']), 'ms')
    if matches['microseconds']:
        delta += np.timedelta64(np.int32(matches['microseconds']), 'us')
    return delta

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
