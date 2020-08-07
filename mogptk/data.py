import csv
import copy
import inspect
import numpy as np
from .bnse import *
from .serie import *
from scipy import signal
import dateutil, datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import re
import logging
from sklearn.mixture import GaussianMixture as gmm

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

logger = logging.getLogger('mogptk')

class TransformDetrend(TransformBase):
    """
    TransformDetrend is a transformer that detrends the data. It uses NumPy `polyfit` to find an `n` degree polynomial that removes the trend.

    Args:
        degree (int): Polynomial degree that will be fit, i.e. `2` will find a quadratic trend and remove it from the data.
    """
    # TODO: add regression?
    def __init__(self, degree=1):
        self.degree = degree

    def set_data(self, data):
        if data.get_input_dims() != 1:
            raise Exception("can only remove ranges on one dimensional input data")

        self.coef = np.polyfit(data.X[0].transformed[data.mask], data.Y.transformed[data.mask], self.degree)
        # reg = Ridge(alpha=0.1, fit_intercept=True)
        # reg.fit(data.X, data.Y)
        # self.trend = reg

    def forward(self, y, x=None):
        return y - np.polyval(self.coef, x[:, 0])
        # return y - self.trend.predict(x)
    
    def backward(self, y, x=None):
        return y + np.polyval(self.coef, x[:, 0])
        # return y + self.trend.predict(x)

class TransformLinear(TransformBase):
    """
    TransformLinear transforms the data linearly so that y => (y-bias)/slope.
    """
    def __init__(self, bias=0.0, slope=1.0):
        self.bias = bias
        self.slope = slope

    def set_data(self, data):
        pass

    def forward(self, y, x=None):
        return (y-self.bias)/self.slope

    def backward(self, y, x=None):
        return self.bias + self.slope*y

class TransformNormalize(TransformBase):
    """
    TransformNormalize is a transformer that normalizes the data so that the y-axis is between -1 and 1.
    """
    def __init__(self):
        pass

    def set_data(self, data):
        self.ymin = np.amin(data.Y.transformed[data.mask])
        self.ymax = np.amax(data.Y.transformed[data.mask])

    def forward(self, y, x=None):
        return -1.0 + 2.0*(y-self.ymin)/(self.ymax-self.ymin)
    
    def backward(self, y, x=None):
        return (y+1.0)/2.0*(self.ymax-self.ymin)+self.ymin

class TransformLog(TransformBase):
    """
    TransformLog is a transformer that takes the log of the data. Data is automatically shifted in the y-axis so that all values are greater than or equal to 1.
    """
    def __init__(self):
        pass

    def set_data(self, data):
        self.shift = 1 - data.Y.transformed.min()
        self.mean = np.log(data.Y.transformed + self.shift).mean()

    def forward(self, y, x=None):
        return np.log(y + self.shift) - self.mean
    
    def backward(self, y, x=None):
        return np.exp(y + self.mean) - self.shift

class TransformWhiten(TransformBase):
    """
    Transform the data so it has mean 0 and variance 1
    """
    def __init__(self):
        pass
    
    def set_data(self, data):
        # take only the non-removed observations
        self.mean = data.Y.transformed[data.mask].mean()
        self.std = data.Y.transformed[data.mask].std()
        
    def forward(self, y, x=None):
        return (y - self.mean) / self.std
    
    def backward(self, y, x=None):
        return (y * self.std) + self.mean

# TODO: add TransformFunction with a given mean function

################################################################
################################################################
################################################################

def LoadFunction(f, start, end, n, var=0.0, name="", random=False):
    """
    LoadFunction loads a dataset from a given function y = f(x) + N(0,var). It will pick n data points between start and end for x, for which f is being evaluated. By default the n points are spread equally over the interval, with random=True they will be picked randomly.

    The function should take one argument x with shape (n,input_dims) and return y with shape (n). If your data has only one input dimension, you can use x[:,0] to select only the first (and only) input dimension.

    Args:
        f (function): Function taking x with shape (n,input_dims) and returning shape (n) as y.
        n (int): Number of data points to pick between start and end.
        start (float, list): Define start of interval.
        end (float, list): Define end of interval.
        var (float, optional): Variance added to the output.
        name (str, optional): Name of data.
        random (boolean): Select points randomly between start and end (defaults to False).

    Returns:
        mogptk.data.Data

    Examples:
        >>> LoadFunction(lambda x: np.sin(3*x[:,0]), 0, 10, n=200, var=0.1, name='Sine wave')
        <mogptk.data.Data at ...>
    """

    if type(start) is not type(end):
        raise ValueError("start and end must be of the same type")
    if isinstance(start, np.ndarray):
        if start.ndim == 0:
            start = [start.item()]
            end = [end.item()]
        else:
            start = list(start)
            end = list(end)
    if not isinstance(start, list):
        start = [start]
        end = [end]

    if len(start) != len(end):
        raise ValueError("start and end must be of the same length")
    for i, j in zip(start, end):
        if type(i) is not type(j):
            raise ValueError("start and end must be of the same type for every pair")

    input_dims = len(start)
    _check_function(f, input_dims)

    x = np.empty((n, input_dims))
    for i in range(input_dims):
        if start[i] >= end[i]:
            if input_dims == 1:
                raise ValueError("start must be lower than end")
            else:
                raise ValueError("start must be lower than end for input dimension %d" % (i))

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

################################################################
################################################################
################################################################

class Data:
    def __init__(self, X, Y, name=None, x_labels=None, y_label=None):
        """
        Data class holds all the observations, latent functions and prediction data.

        This class takes the data raw, but you can load data also conveniently using
        LoadFunction, LoadCSV, LoadDataFrame, etc. This class allows to modify the data before being passed into the model.
        Examples are transforming data, such as detrending or taking the log, removing data range to simulate sensor failure,
        and aggregating data for given spans on X, such as aggregating daily data into
        weekly data. Additionally, we also use this class to set the range we want to predict.

        It is possible to use the format given by np.meshgrid for X and its values in Y.

        Args:
            X (list, numpy.ndarray, dict): Independent variable data of shape (n) or (n,input_dims).
            Y (list, numpy.ndarray): Dependent variable data of shape (n).
            name (str, optional): Name of data.
            x_labels (str, list of str, optional): Name or names of input dimensions.
            y_label (str, optional): Name of output dimension.

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
                X = list(map(list, zip(*[X[key] for key in x_labels])))

        # check if X and Y are correct inputs
        if isinstance(X, list):
            if all(isinstance(x, list) for x in X):
                m = len(X[0])
                if not all(len(x) == m for x in X[1:]):
                    raise ValueError("X list items must all be lists of the same length")
                if not all(all(isinstance(val, (int, float)) for val in x) for x in X):
                    raise ValueError("X list items must all be lists of numbers")
            elif all(isinstance(x, np.ndarray) for x in X):
                m = len(X[0])
                if not all(len(x) == m for x in X[1:]):
                    raise ValueError("X list items must all be numpy.ndarrays of the same length")
            elif not all(isinstance(x, (int, float)) for x in X):
                raise ValueError("X list items must be all lists, all numpy.ndarrays, or all numbers")
            X = np.array(X)
        if isinstance(Y, list):
            if not all(isinstance(y, (int, float)) for y in Y):
                raise ValueError("Y list items must all be numbers")
            Y = np.array(Y)
        if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
            raise ValueError("X and Y must be lists or numpy arrays, if dicts are passed then x_labels and/or y_label must also be set")
        
        # try to cast unknown data types
        if not np.issubdtype(X.dtype, np.number) and not np.issubdtype(X.dtype, np.datetime64):
            try:
                X = X.astype(np.float)
            except:
                try:
                    X = X.astype(np.datetime64)
                except:
                    raise ValueError("X data must have a number or datetime64 data type")
        if not np.issubdtype(Y.dtype, np.number):
            try:
                Y = Y.astype(np.float)
            except:
                raise ValueError("Y data must have a number data type")

        # convert X datetime64[us] to a higher unit like s, m, h, D, ...
        if np.issubdtype(X.dtype, np.datetime64):
            X = _datetime64_to_higher_unit(X)

        # convert meshgrids to flat arrays
        if 2 < X.ndim and 1 < Y.ndim and X.shape[1:] == Y.shape:
            X = np.vstack(list(map(np.ravel, X))).T
            Y = np.ravel(Y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError("X must be either a one or two dimensional array of data")
        if Y.ndim != 1:
            raise ValueError("Y must be a one dimensional array of data")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must be of the same length")
        if Y.shape[0] == 0:
            raise ValueError("X and Y must have a length greater than zero")

        input_dims = X.shape[1]
        self.X = [Serie(X[:,i]) for i in range(input_dims)] # [shape (n)] * input_dims
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

        self.name = ''
        if isinstance(name, str):
            self.name = name
        elif isinstance(y_label, str):
            self.name = y_label

        self.Y_label = 'Y'
        if isinstance(y_label, str):
            self.Y_label = y_label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        df = pd.DataFrame()
        for i in range(len(self.X)):
            df[self.X_labels[i]] = self.X[i]
        df[self.Y_label] = self.Y
        return repr(df)

    def copy(self):
        """
        Make a deep copy of Data.

        Returns:
            mogptk.data.Data

        Examples:
            >>> other = data.copy()
        """
        return copy.deepcopy(self)

    def set_name(self, name):
        """
        Set name for data.

        Args:
            name (str): Name of data.

        Examples:
            >>> data.set_name('Channel A')
        """
        self.name = name

    def set_labels(self, x_labels, y_label):
        """
        Set axes labels for plots.

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
        Set a (latent) function for the data, ie. the theoretical or true signal. This is used for plotting purposes and is optional.
    
        The function should take one argument x with shape (n,input_dims) and return y with shape (n). If your data has only one input dimension, you can use x[:,0] to select only the first (and only) input dimension.

        Args:
            f (function): Function taking x with shape (n,input_dims) and returning shape (n) as y.

        Examples:
            >>> data.set_function(lambda x: np.sin(3*x[:,0])
        """
        _check_function(f, self.get_input_dims())
        self.F = f

    def transform_x(self, transformer):
        """
        Transform the X axis data by using one of the provided transformers, such as `TransformDetrend`, `TransformLinear`, `TransformLog`, `TransformNormalize`, `TransformWhiten`, ...

        Args:
            transformer (obj, list of obj): Transformer objects derived from TransformBase for each input dimension. A single object is applied to all input dimensions.

        Examples:
            >>> data.transform_x(mogptk.TransformLinear(5.0, 20.0))
        """

        if not isinstance(transformer, list):
            transformer = [transformer] * self.get_input_dims()
        elif len(transformer) != self.get_input_dims():
            raise ValueError('transformer must be a list of transformers for each input dimension')

        for i, t in enumerate(transformer):
            if isinstance(t, type):
                t = t()
            t.set_data(self)
            self.X[i].apply(t)
    
    def rescale_x(self):
        """
        Rescale the X axis so that it is between 0 and 1000 internally. This can help when the range of your x-axis is much smaller or bigger than a 1000, which affects effectiveness of training.

        Examples:
            >>> data.rescale_x()
        """
        xmin = [np.min(self.X[i].get_transformed()) for i in range(self.get_input_dims())]
        xmax = [np.max(self.X[i].get_transformed()) for i in range(self.get_input_dims())]
        transforms = [TransformLinear(xmin[i], (xmax[i]-xmin[i])/1000.0) for i in range(self.get_input_dims())]
        self.transform_x(transforms)

    def transform(self, transformer):
        """
        Transform the Y axis data by using one of the provided transformers, such as `TransformDetrend`, `TransformLinear`, `TransformLog`, `TransformNormalize`, `TransformWhiten`, ...

        Args:
            transformer (obj): Transformer object derived from TransformBase.

        Examples:
            >>> data.transform(mogptk.TransformDetrend(degree=2))        # remove polynomial trend
            >>> data.transform(mogptk.TransformLinear(slope=1, bias=2))  # remove linear trend
            >>> data.transform(mogptk.TransformLog)                      # log transform the data
            >>> data.transform(mogptk.TransformNormalize)                # transform to [-1,1]
            >>> data.transform(mogptk.TransformWhiten)                   # transform to mean=0, var=1
        """

        t = transformer
        if isinstance(t, type):
            t = transformer()
        t.set_data(self)

        X = np.array([x.transformed for x in self.X]).T
        self.Y.apply(t, X)
    
    def filter(self, start, end):
        """
        Filter the data range to be between start and end.

        Args:
            start (float, str): Start of interval.
            end (float, str): End of interval.

        Examples:
            >>> data = mogptk.LoadFunction(lambda x: np.sin(3*x[:,0]), 0, 10, n=200, var=0.1, name='Sine wave')
            >>> data.filter(3, 8)
        
            >>> data = mogptk.LoadCSV('gold.csv', 'Date', 'Price')
            >>> data.filter('2016-01-15', '2016-06-15')
        """
        if self.get_input_dims() != 1:
            raise ValueError("can only filter on one dimensional input data")

        start = self._normalize_val(start)
        end = self._normalize_val(end)
        
        ind = (self.X[0] >= start[0]) & (self.X[0] < end[0])

        self.X[0] = self.X[0][ind]
        self.Y = self.Y[ind]
        self.mask = self.mask[ind]

    def aggregate(self, duration, f=np.mean):
        """
        Aggregate the data by duration and apply a function to obtain a reduced dataset.

        For example, group daily data by week and take the mean.
        The duration can be set as a number which defined the intervals on the X axis,
        or by a string written in the duration format with:
        y=year, M=month, w=week, d=day, h=hour, m=minute, and s=second.
        For example, 3w1d means three weeks and one day, ie. 22 days, or 6M to mean six months.

        Args:
            duration (float, str): Duration along the X axis or as a string in the duration format.
            f (function, optional): Function to use to reduce data, by default uses np.mean.

        Examples:
            >>> data.aggregate(5)

            >>> data.aggregate('2w', f=np.sum)
        """
        if self.get_input_dims() != 1:
            raise ValueError("can only aggregate on one dimensional input data")
        
        start = np.min(self.X[0])
        end = np.max(self.X[0])
        step = _parse_delta(duration)

        X = np.arange(start+step/2, end+step/2, step)
        Y = np.empty((len(X)))
        for i in range(len(X)):
            ind = (self.X[0] >= X[i]-step/2) & (self.X[0] < X[i]+step/2)
            Y[i] = f(self.Y[ind])

        self.X = [Serie(X, self.X[0].transformers)]
        self.Y = Serie(Y, self.Y.transformers, np.array([x.transformed for x in self.X]).T)
        self.mask = np.array([True] * len(self.X[0]))

    ################################################################

    def get_name(self):
        """
        Return the name.

        Returns:
            str.

        Examples:
            >>> data.get_name()
            'A'
        """
        return self.name

    def has_test_data(self):
        """
        Returns True if observations have been removed using the remove_* methods.

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
            int: Input dimensions.

        Examples:
            >>> data.get_input_dims()
            2
        """
        return len(self.X)
    
    def get_data(self):
        """
        Returns all observations, train and test.

        Returns:
            numpy.ndarray: X data of shape (n,input_dims).
            numpy.ndarray: Y data of shape (n).

        Examples:
            >>> x, y = data.get_data()
        """
        return np.array([x for x in self.X]).T, self.Y

    def get_train_data(self):
        """
        Returns the observations used for training.

        Returns:
            numpy.ndarray: X data of shape (n,input_dims).
            numpy.ndarray: Y data of shape (n).

        Examples:
            >>> x, y = data.get_train_data()
        """
        return np.array([x[self.mask] for x in self.X]).T, self.Y[self.mask]

    def get_test_data(self):
        """
        Returns the observations used for testing.

        Returns:
            numpy.ndarray: X data of shape (n,input_dims).
            numpy.ndarray: Y data of shape (n).

        Examples:
            >>> x, y = data.get_test_data()
        """
        return np.array([x[~self.mask] for x in self.X]).T, self.Y[~self.mask]

    ################################################################
    
    def remove_randomly(self, n=None, pct=None):
        """
        Removes observations randomly on the whole range. Either 'n' observations are removed, or a percentage of the observations.

        Args:
            n (int, optional): Number of observations to remove randomly.
            pct (float, optional): Percentage in interval [0,1] of observations to remove randomly.

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
    
    def remove_range(self, start=None, end=None):
        """
        Removes observations in the interval [start,end].
        
        Args:
            start (float, str, optional): Start of interval. Defaults to first value in observations.
            end (float, str, optional): End of interval. Defaults to last value in observations.

        Examples:
            >>> data = mogptk.LoadFunction(lambda x: np.sin(3*x[:,0]), 0, 10, n=200, var=0.1, name='Sine wave')
            >>> data.remove_range(3, 8)
        
            >>> data = mogptk.LoadCSV('gold.csv', 'Date', 'Price')
            >>> data.remove_range('2016-01-15', '2016-06-15')
        """
        if self.get_input_dims() != 1:
            raise Exception("can only remove ranges on one dimensional input data")

        if start is None:
            start = np.min(self.X[0])
        if end is None:
            end = np.max(self.X[0])

        start = self._normalize_val(start)
        end = self._normalize_val(end)

        idx = np.where(np.logical_and(self.X[0] >= start[0], self.X[0] <= end[0]))
        self.mask[idx] = False
        self.removed_ranges[0].append([start[0], end[0]])
    
    def remove_relative_range(self, start=0.0, end=1.0):
        """
        Removes observations between start and end as a percentage of the number of observations. So '0' is the first observation, '0.5' is the middle observation, and '1' is the last observation.

        Args:
            start (float): Start percentage in interval [0,1].
            end (float): End percentage in interval [0,1].
        """
        if self.get_input_dims() != 1:
            raise Exception("can only remove ranges on one dimensional input data")

        start = self._normalize_val(start)
        end = self._normalize_val(end)

        x_min = np.min(self.X[0])
        x_max = np.max(self.X[0])
        for i in range(self.get_input_dims()):
            start[i] = x_min + max(0.0, min(1.0, start[i])) * (x_max-x_min)
            end[i] = x_min + max(0.0, min(1.0, end[i])) * (x_max-x_min)

        idx = np.where(np.logical_and(self.X[0] >= start[0], self.X[0] <= end[0]))
        self.mask[idx] = False
        self.removed_ranges[0].append([start[0], end[0]])

    def remove_random_ranges(self, n, duration):
        """
        Removes a number of ranges to simulate sensor failure. May remove fewer ranges if there is no more room to remove a range in the remaining data.

        Args:
            n (int): Number of ranges to remove.
            duration (float, str): Width of ranges to remove, can use a number or the duration format syntax (see aggregate()).

        Examples:
            >>> data.remove_random_ranges(2, 5) # remove two ranges that are 5 wide in input space

            >>> data.remove_random_ranges(3, '1d') # remove three ranges that are 1 day wide
        """
        if self.get_input_dims() != 1:
            raise Exception("can only remove ranges on one dimensional input data")
        if n < 1:
            return

        delta = _parse_delta(duration)
        m = (np.max(self.X[0])-np.min(self.X[0])) - n*delta
        if m <= 0:
            raise Exception("no data left after removing ranges")

        locs = self.X[0] <= (np.max(self.X[0])-delta)
        locs[sum(locs)] = True # make sure the last data point can be deleted
        for i in range(n):
            if len(self.X[0][locs]) == 0:
                break # range could not be removed, there is no remaining data range of width delta
            x = self.X[0][locs][np.random.randint(len(self.X[0][locs]))]
            locs[(self.X[0] > x-delta) & (self.X[0] < x+delta)] = False
            self.mask[(self.X[0] >= x) & (self.X[0] < x+delta)] = False
            self.removed_ranges[0].append([x, x+delta])

    def remove_index(self, index):
        """
        Removes observations of given index

        Args:
            index(array-like): Array of indexes of the data to remove.
        """
        if isinstance(index, list):
            index = np.array(index)
        elif not isinstance(index, np.ndarray):
            raise ValueError("index must be list or numpy array")

        self.mask[index] = False
    
    ################################################################
    
    def get_prediction(self, name, sigma=2):
        """
        Returns the prediction of a given name with a normal variance of sigma.

        Args:
            name (str): Name of the prediction, equals the name of the model that made the prediction.
            sigma (float): The uncertainty interval calculated at mean-sigma*var and mean+sigma*var. Defaults to 2,

        Returns:
            numpy.ndarray: X prediction of shape (n,input_dims).
            numpy.ndarray: Y mean prediction of shape (n,).
            numpy.ndarray: Y lower prediction of uncertainty interval of shape (n,).
            numpy.ndarray: Y upper prediction of uncertainty interval of shape (n,).

        Examples:
            >>> x, y_mean, y_var_lower, y_var_upper = data.get_prediction('MOSM', sigma=1)
        """
        if name not in self.Y_mu_pred:
            raise Exception("prediction name '%s' does not exist" % (name))
       
        mu = self.Y_mu_pred[name]
        lower = mu - sigma * np.sqrt(self.Y_var_pred[name])
        upper = mu + sigma * np.sqrt(self.Y_var_pred[name])

        X_pred = np.array([x.transformed for x in self.X_pred]).T
        mu = self.Y.detransform(mu, X_pred)
        lower = self.Y.detransform(lower, X_pred)
        upper = self.Y.detransform(upper, X_pred)
        return np.array([x for x in self.X_pred]).T, mu, lower, upper

    def set_prediction_range(self, start=None, end=None, n=None, step=None):
        """
        Sets the prediction range. The interval is set with [start,end], with either 'n' points or a
        given 'step' between the points.

        Args:
            start (float, str, optional): Start of interval, defaults to the first observation.
            end (float, str, optional): End of interval, defaults to the last observation.
            n (int, optional): Number of points to generate in the interval.
            step (float, str, optional): Spacing between points in the interval.

            If neither 'step' or 'n' is passed, default number of points is 100.

        Examples:
            >>> data = mogptk.LoadFunction(lambda x: np.sin(3*x[:,0]), 0, 10, n=200, var=0.1, name='Sine wave')
            >>> data.set_prediction_range(3, 8, 200)
        
            >>> data = mogptk.LoadCSV('gold.csv', 'Date', 'Price')
            >>> data.set_prediction_range('2016-01-15', '2016-06-15', step='1d')
        """
        if self.get_input_dims() != 1:
            raise Exception("can only set prediction range on one dimensional input data")

        if start is None:
            start = [x[0] for x in self.X]
        if end is None:
            start = [x[-1] for x in self.X]
        
        start = self._normalize_val(start)
        end = self._normalize_val(end)

        # TODO: works for multi input dims?
        if end <= start:
            raise ValueError("start must be lower than end")

        # TODO: prediction range for multi input dimension; fix other axes to zero so we can plot?
        X_pred = [np.array([])] * self.get_input_dims()
        if step is None and n is not None:
            for i in range(self.get_input_dims()):
                X_pred[i] = np.linspace(start[i], end[i], n)
        else:
            if self.get_input_dims() != 1:
                raise ValueError("cannot use step for multi dimensional input, use n")
            if step is None:
                step = (end[0]-start[0])/100
            else:
                step = _parse_delta(step)
            X_pred[0] = np.arange(start[0], end[0]+step, step)
        
        self.X_pred = [Serie(x, self.X[i].transformers) for i, x in enumerate(X_pred)]
    
    def set_prediction_x(self, X):
        """
        Set the prediction range directly.

        Args:
            X (list, numpy.ndarray): Array of shape (n) or (n,input_dims) with input values to predict at.

        Examples:
            >>> data.set_prediction_x([5.0, 5.5, 6.0, 6.5, 7.0])
        """
        if isinstance(X, list):
            X = np.array(X)
        elif not isinstance(X, np.ndarray):
            raise ValueError("X expected to be a list or numpy.ndarray")

        X = X.astype(np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2 or X.shape[1] != self.get_input_dims():
            raise ValueError("X shape must be (n,input_dims)")

        self.X_pred = [Serie(X[:,i], self.X[i].transformers) for i in range(self.get_input_dims())]

        # clear old prediction data now that X_pred has been updated
        self.Y_mu_pred = {}
        self.Y_var_pred = {}

    def set_prediction(self, name, mu, var):
        self.Y_mu_pred[name] = mu
        self.Y_var_pred[name] = var

    ################################################################

    def get_nyquist_estimation(self):
        """
        Estimate nyquist frequency by taking 0.5/(minimum distance of points).

        Returns:
            numpy.ndarray: Nyquist frequency array of shape (input_dims,).

        Examples:
            >>> freqs = data.get_nyquist_estimation()
        """
        input_dims = self.get_input_dims()

        nyquist = np.empty((input_dims))
        for i in range(self.get_input_dims()):
            x = np.sort(self.X[i].transformed[self.mask])
            dist = np.abs(x[1:]-x[:-1])
            dist = np.min(dist[np.nonzero(dist)])
            nyquist[i] = 0.5/dist
        return nyquist

    def get_bnse_estimation(self, Q=1, n=8000):
        """
        Peaks estimation using BNSE (Bayesian Non-parametric Spectral Estimation).

        Args:
            Q (int): Number of peaks to find, defaults to 1.
            n (int): Number of points of the grid to evaluate frequencies, defaults to 5000.

        Returns:
            numpy.ndarray: Amplitude array of shape (input_dims,Q).
            numpy.ndarray: Frequency array of shape (input_dims,Q) in radians.
            numpy.ndarray: Variance array of shape (input_dims,Q) in radians.

        Examples:
            >>> amplitudes, means, variances = data.get_bnse_estimation()
        """
        input_dims = self.get_input_dims()

        # Gaussian: f(x) = A * exp((x-B)^2 / (2C^2))
        # Ie. A is the amplitude or peak height, B the mean or peak position, and C the variance or peak width
        A = np.zeros((input_dims, Q))
        B = np.zeros((input_dims, Q))
        C = np.zeros((input_dims, Q))

        nyquist = self.get_nyquist_estimation()
        for i in range(input_dims):
            x = np.array(self.X[i].transformed[self.mask])
            y = np.array(self.Y.transformed[self.mask])
            bnse = bse(x, y)
            bnse.set_freqspace(nyquist[i], dimension=n)
            bnse.train()
            bnse.compute_moments()

            amplitudes, positions, variances = bnse.get_freq_peaks()

            # TODO: sqrt of amplitudes? vs LS?
            if len(positions) == 0:
                continue

            n_pos = len(positions)
            if n_pos < Q and n_pos != 0:
                # if there not enough peaks, we will repeat them
                j = 0
                while len(positions) < Q:
                    amplitudes = np.append(amplitudes, amplitudes[j])
                    positions = np.append(positions, positions[j])
                    variances = np.append(variances, variances[j])
                    j = (j+1) % n_pos

            A[i,:] = amplitudes[:Q]
            B[i,:] = positions[:Q]
            C[i,:] = variances[:Q]

        return A, B, C

    def get_lombscargle_estimation(self, Q=1, n=50000):
        """
        Peak estimation using Lomb Scargle.

        Args:
            Q (int): Number of peaks to find, defaults to 1.
            n (int): Number of points to use for Lomb Scargle, defaults to 50000.

        Returns:
            numpy.ndarray: Amplitude array of shape (input_dims,Q).
            numpy.ndarray: Frequency array of shape (input_dims,Q) in radians.
            numpy.ndarray: Variance array of shape (input_dims,Q) in radians.

        Examples:
            >>> amplitudes, means, variances = data.get_lombscargle_estimation()
        """
        input_dims = self.get_input_dims()

        # Gaussian: f(x) = A * exp((x-B)^2 / (2C^2))
        # Ie. A is the amplitude or peak height, B the mean or peak position, and C the variance or peak width
        A = np.zeros((input_dims, Q))
        B = np.zeros((input_dims, Q))
        C = np.zeros((input_dims, Q))

        nyquist = self.get_nyquist_estimation() * 2 * np.pi
        for i in range(input_dims):
            freq = np.linspace(0, nyquist[i], n+1)[1:]
            dfreq = freq[1]-freq[0]

            y = signal.lombscargle(self.X[i].transformed[self.mask], self.Y.transformed[self.mask], freq)
            ind, _ = signal.find_peaks(y)
            ind = ind[np.argsort(y[ind])[::-1]] # sort by biggest peak first

            widths, width_heights, _, _ = signal.peak_widths(y, ind, rel_height=0.5)
            widths *= dfreq / np.pi / 2.0

            positions = freq[ind] / np.pi / 2.0
            amplitudes = y[ind]
            variances = widths / np.sqrt(8 * np.log(amplitudes / width_heights)) # from full-width half-maximum to Gaussian sigma

            n_pos = len(positions)
            if n_pos < Q and n_pos != 0:
                # if there not enough peaks, we will repeat them
                j = 0
                while len(positions) < Q:
                    amplitudes = np.append(amplitudes, amplitudes[j])
                    positions = np.append(positions, positions[j])
                    variances = np.append(variances, variances[j])
                    j = (j+1) % n_pos

            A[i,:] = amplitudes[:Q]
            B[i,:] = positions[:Q]
            C[i,:] = variances[:Q]

        return A, B, C

    def gmm_estimation(self, Q=1, n=50000):
        """
        Parameter estimation using a GMM on a PSD estimate

        Args:
            Q (int): Number of peaks to find, defaults to 1.
            n (int): Number of points to use for Lomb Scargle, defaults to 50000.

        Returns:
            numpy.ndarray: Amplitude array of shape (input_dims,Q).
            numpy.ndarray: Frequency array of shape (input_dims,Q) in radians.
            numpy.ndarray: Variance array of shape (input_dims,Q) in radians.

        Examples:
            >>> amplitudes, means, variances = data.gmm_init()
        """

        input_dims = self.get_input_dims()
        
        amplitudes = np.zeros((input_dims, Q))
        means = np.zeros((input_dims, Q))
        variances = np.zeros((input_dims, Q))

        nyquist = self.get_nyquist_estimation()
        
        for i in range(input_dims):
            freqs = np.linspace(0, nyquist[i], n+1)[1:]
            
            psd = signal.lombscargle(self.X[i].transformed[self.mask], self.Y.transformed[self.mask], freqs, normalize=True)
            
            ind, _ = signal.find_peaks(psd)
            
            # sort by biggest peak first
            ind = ind[np.argsort(psd[ind])[::-1]]
            
            model = gmm(
                Q,
                covariance_type='diag',
                max_iter=500,
                tol=1e-5,
                means_init=freqs[ind][:Q].reshape(-1, 1),
            ).fit(freqs.reshape(-1, 1), psd)
            
            variances[i, :] = model.covariances_[:, 0]
            means[i, :] = (model.means_ / (2 * np.pi))[:, 0]
            amplitudes[i, :] = model.weights_ / np.sqrt(2 * np. pi * variances[i, :])
            
        return variances, means, amplitudes

    def plot(self, ax=None, legend=True, transformed=False):
        """
        Plot the data including removed observations, latent function, and predictions.

        Args:
            ax (matplotlib.axes.Axes, optional): Draw to this axes, otherwise draw to the current axes.
            legend (boolean, optional): Display legend.
            transformed (boolean, optional): Display transformed Y data as used for training.

        Returns:
            matplotlib.axes.Axes

        Examples:
            >>> ax = data.plot()
        """
        # TODO: ability to plot conditional or marginal distribution to reduce input dims
        if self.get_input_dims() > 2:
            raise Exception("cannot plot more than two input dimensions")
        if self.get_input_dims() == 2:
            raise Exception("two dimensional input data not yet implemented") # TODO

        if ax is None:
            ax = plt.gca()

        legends = []
        colors = list(matplotlib.colors.TABLEAU_COLORS)
        for i, name in enumerate(self.Y_mu_pred):
            if self.Y_mu_pred[name].size != 0:
                if transformed:
                    mu = self.Y_mu_pred[name].get_transformed()
                    var = self.Y_var_pred[name].get_transformed()
                else:
                    mu = self.Y_mu_pred[name]
                    var = self.Y_var_pred[name]

                lower = mu - var
                upper = mu + var

                idx = np.argsort(self.X_pred[0])
                ax.plot(self.X_pred[0][idx], mu[idx], ls='-', color=colors[i], lw=2)
                ax.fill_between(self.X_pred[0][idx], lower[idx], upper[idx], color=colors[i], alpha=0.1)
                ax.plot(self.X_pred[0][idx], lower[idx], ls='-', color=colors[i], lw=1, alpha=0.5)
                ax.plot(self.X_pred[0][idx], upper[idx], ls='-', color=colors[i], lw=1, alpha=0.5)

                label = 'Prediction ' + name
                legends.append(plt.Line2D([0], [0], ls='-', color=colors[i], lw=2, label=label))

        if self.F is not None:
            n = len(self.X[0])*10
            x_min = np.min(self.X[0])
            x_max = np.max(self.X[0])
            if len(self.X_pred[0]) != 0:
                x_min = min(x_min, np.min(self.X_pred[0]))
                x_max = max(x_max, np.max(self.X_pred[0]))

            x = np.empty((n, 1))
            x[:,0] = np.linspace(x_min, x_max, n)
            y = self.F(x)
            if transformed:
                y = self.Y.transform(y)

            ax.plot(x[:,0], y, 'r--', lw=1)
            legends.append(plt.Line2D([0], [0], ls='--', color='r', label='True'))

        Y = self.Y
        if transformed:
            Y = Y.get_transformed()
        idx = np.argsort(self.X[0])
        ax.plot(self.X[0][idx], Y[idx], 'k--', alpha=0.8)
        legends.append(plt.Line2D([0], [0], ls='--', color='k', label='All Points'))

        x, y = self.get_train_data()
        ax.plot(x[:,0], y, 'k.', mew=1, ms=13, markeredgecolor='white')
        legends.append(plt.Line2D([0], [0], ls='', marker='.', color='k', mew=1, ms=13, label='Training Points'))

        if self.has_test_data():
            for removed_range in self.removed_ranges[0]:
                x0 = removed_range[0]
                x1 = removed_range[1]
                y0 = ax.get_ylim()[0]
                y1 = ax.get_ylim()[1]
                ax.add_patch(patches.Rectangle(
                    (x0, y0), x1-x0, y1-y0, fill=True, color='xkcd:strawberry', alpha=0.25, lw=0,
                ))
            legends.append(patches.Rectangle(
                (1, 1), 1, 1, fill=True, color='xkcd:strawberry', alpha=0.5, lw=0, label='Removed Ranges'
            ))

        xmin = self.X[0].min()
        xmax = self.X[0].max()
        ax.set_xlim(xmin - (xmax - xmin)*0.001, xmax + (xmax - xmin)*0.001)

        ax.set_xlabel(self.X_labels[0])
        ax.set_ylabel(self.Y_label)
        ax.set_title(self.name, fontsize=14)

        if 0 < len(legends) and legend:
            ax.legend(handles=legends, loc='upper center', ncol=len(legends), bbox_to_anchor=(0.5, 1.5))
        return ax

    def plot_spectrum(self, method='lombscargle', ax=None, per=None, maxfreq=None, transformed=False):
        """
        Plot the spectrum of the data.

        Args:
            method (str, optional): Set the method to get the spectrum: lombscargle or bnse.
            ax (matplotlib.axes.Axes, optional): Draw to this axes, otherwise draw to the current axes.
            per (str, float, np.timedelta64, optional): Set the scale of the X axis depending on the formatter used, eg. per=5, per='day', or per='3d'.
            maxfreq (float, optional): Maximum frequency to plot, otherwise the Nyquist frequency is used.
            transformed (boolean, optional): Display transformed Y data as used for training.

        Returns:
            matplotlib.axes.Axes

        Examples:
            >>> ax = data.plot_spectrum(method='bnse')
        """
        # TODO: ability to plot conditional or marginal distribution to reduce input dims
        if self.get_input_dims() > 2:
            raise Exception("cannot plot more than two input dimensions")
        if self.get_input_dims() == 2:
            raise Exception("two dimensional input data not yet implemented") # TODO

        if ax is None:
            ax = plt.gca()
        
        ax.set_title(self.name, fontsize=36)

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
        idx = np.argsort(X)
        X = X[idx] * X_scale
        Y = self.Y[idx]
        if transformed:
            Y = Y.get_transformed()

        nyquist = maxfreq
        if nyquist is None:
            dist = np.abs(X[1:]-X[:-1])
            nyquist = 0.5 / np.average(dist)

        X_freq = np.linspace(0.0, nyquist, 10001)[1:]
        Y_freq_err = []
        if method == 'lombscargle':
            Y_freq = signal.lombscargle(X*2.0*np.pi, Y, X_freq)
        elif method == 'bnse':
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

        ax.plot(X_freq, Y_freq, '-', color='xkcd:strawberry', lw=2.3)
        if len(Y_freq_err) != 0:
            ax.fill_between(X_freq, Y_freq-Y_freq_err, Y_freq+Y_freq_err, alpha=0.4)
        ax.set_title(self.name + ' Spectrum', fontsize=14)

        xmin = X_freq.min()
        xmax = X_freq.max()
        ax.set_xlim(xmin - (xmax - xmin)*0.005, xmax + (xmax - xmin)*0.005)
        ax.set_yticks([])
        ax.set_ylim(0, None)
        return ax

    def _normalize_val(self, val):
        if val is None:
            return val
        if isinstance(val, np.ndarray):
            if val.ndim == 0:
                val = [val.item()]
            else:
                val = list(val)
        elif not isinstance(val, list):
            val = [val] * self.get_input_dims()
        if len(val) != self.get_input_dims():
            raise ValueError("value must be a scalar or a list of values for each input dimension")

        for i in range(self.get_input_dims()):
            try:
                val[i] = self.X[i].dtype.type(val[i])
            except:
                pass
        return val

def _check_function(f, input_dims):
    if not inspect.isfunction(f):
        raise ValueError("function must take X as a parameter")

    sig = inspect.signature(f)
    if not len(sig.parameters) == 1:
        raise ValueError("function must take X as a parameter")

    x = np.ones((1, input_dims))
    y = f(x)
    if len(y.shape) != 1 or y.shape[0] != 1:
        raise ValueError("function must return Y with shape (n), note that X has shape (n,input_dims)")

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
