import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .data import Data, _is_iterable

def LoadCSV(filename, x_col=0, y_col=1, y_err_col=None, name=None, **kwargs):
    """
    LoadCSV loads a dataset from a given CSV file. It loads in `x_col` as the names of the input dimension columns, and `y_col` as the names of the output columns.

    Args:
        filename (str): CSV filename.
        x_col (int, str, list of int or str): Names or indices of X column(s) in CSV.
        y_col (int, str, list of int or str): Names or indices of Y column(s) in CSV.
        name (str, list): Name or names of data channels.
        **kwargs: Additional keyword arguments for csv.DictReader.

    Returns:
        mogptk.data.Data or mogptk.dataset.DataSet

    Examples:
        >>> LoadCSV('gold.csv', 'Date', 'Price', name='Gold')
        <mogptk.dataset.DataSet at ...>
        >>> LoadCSV('gold.csv', 'Date', 'Price', sep=' ', quotechar='|')
        <mogptk.dataset.DataSet at ...>
    """

    df = pd.read_csv(filename, **kwargs)

    return LoadDataFrame(df, x_col, y_col, y_err_col, name)

def LoadDataFrame(df, x_col=0, y_col=1, y_err_col=None, name=None):
    """
    LoadDataFrame loads a DataFrame from pandas. It loads in `x_col` as the names of the input dimension columns, and `y_col` the names of the output columns.

    Args:
        df (pandas.DataFrame): The pandas DataFrame.
        x_col (int, str, list of int or str): Names or indices of X column(s) in DataFrame.
        y_col (int, str, list of int or str): Names or indices of Y column(s) in DataFrame.
        y_err_col (int, str, list of int or str): Names or indices of Y error column(s) in DataFrame.
        name (str, list of str): Name or names of data channels.

    Returns:
        mogptk.data.Data or mogptk.dataset.DataSet

    Examples:
        >>> df = pd.DataFrame(...)
        >>> LoadDataFrame(df, 'Date', 'Price', name='Gold')
        <mogptk.dataset.DataSet at ...>
    """

    if _is_iterable(x_col):
        x_col = list(x_col)
    if _is_iterable(y_col):
        y_col = list(y_col)
    if (not isinstance(x_col, list) or not all(isinstance(item, int) for item in x_col) and not all(isinstance(item, str) for item in x_col)) and not isinstance(x_col, int) and not isinstance(x_col, str):
        raise ValueError("x_col must be integer, string or list of integers or strings")
    if (not isinstance(y_col, list) or not all(isinstance(item, int) for item in y_col) and not all(isinstance(item, str) for item in y_col)) and not isinstance(y_col, int) and not isinstance(y_col, str):
        raise ValueError("y_col must be integer, string or list of integers or strings")
    if not isinstance(x_col, list):
        x_col = [x_col]
    if not isinstance(y_col, list):
        y_col = [y_col]

    if y_err_col is not None:
        if _is_iterable(y_err_col):
            y_col = list(y_err_col)
        if (not isinstance(y_err_col, list) or not all(isinstance(item, int) for item in y_err_col) and not all(isinstance(item, str) for item in y_err_col)) and not isinstance(y_err_col, int) and not isinstance(y_err_col, str):
            raise ValueError("y_err_col must be integer, string or list of integers or strings")
        if not isinstance(y_err_col, list):
            y_err_col = [y_err_col]
        if len(y_col) != len(y_err_col):
            raise ValueError("y_err_col and y_col must be of the same length")

    if name is None:
        name = [None] * len(y_col)
    else:
        if _is_iterable(name):
            name = list(name)
        else:
            name = [name]
        if len(y_col) != len(name):
            raise ValueError("y_col and name must be of the same length")

    # if columns are indices, convert to column names
    if all(isinstance(item, int) for item in x_col):
        x_col = [df.columns[item] for item in x_col]
    if all(isinstance(item, int) for item in y_col):
        y_col = [df.columns[item] for item in y_col]
    if y_err_col is not None and all(isinstance(item, int) for item in y_err_col):
        y_err_col = [df.columns[item] for item in y_err_col]

    cols = x_col + y_col
    if y_err_col is not None:
        cols += y_err_col
    df = df[cols]
    if len(df.index) == 0:
        raise ValueError("dataframe cannot be empty")

    input_dims = len(x_col)
    x_data = df[x_col]
    x_labels = [str(item) for item in x_col]

    dataset = DataSet()
    for i in range(len(y_col)):
        cols = x_col + [y_col[i]]
        if y_err_col is not None:
            cols += [y_err_col[i]]
        channel = df[cols].dropna()

        y_err = None
        if y_err_col is not None:
            y_err = channel[y_err_col[i]].values

        dataset.append(Data(
            channel[x_col].values,
            channel[y_col[i]].values,
            Y_err=y_err,
            name=name[i],
            x_labels=x_labels,
            y_label=str(y_col[i]),
        ))
    if dataset.get_output_dims() == 1:
        return dataset[0]
    return dataset

################################################################
################################################################
################################################################

class DataSet:
    """
    DataSet is a class that holds multiple Data objects as channels. It is the complete representation of the data used for fitting multi-output Gaussian processes.

    Args:
        *args (mogptk.data.Data, mogptk.dataset.DataSet, list, dict, numpy.ndarray): Accepts multiple arguments, each of which should be either a `DataSet` or `Data` object, a list of `Data` objects or a dictionary of `Data` objects. Each `Data` object will be added to the list of channels. In case of a dictionary, the key will set the name of the channel. If a `DataSet` is passed its channels will be added. It is also possible to pass X and Y data array directly by either passing two `numpy.ndarrays` or two lists of `numpy.ndarrays` for X and Y data.

    Examples:
        Different ways to initiate a DataSet:
        >>> wind_velocity = mogptk.LoadDataFrame(df, x_col='Date', y_col='Wind Velocity', name='wind')
        >>> tidal_height = mogptk.LoadDataFrame(df, x_col='Date', y_col='Tidal Height', name='tidal')
        >>> dataset = mogptk.DataSet(wind_velocity, tidal_height)

        >>> dataset = mogptk.DataSet(
        >>>     mogptk.LoadDataFrame(df, x_col='Date', y_col='Wind Velocity', name='wind'),
        >>>     mogptk.LoadDataFrame(df, x_col='Date', y_col='Tidal Height', name='tidal'),
        >>> )

        >>> dataset = mogptk.DataSet()
        >>> dataset.append(mogptk.LoadDataFrame(df, x_col='Date', y_col='Wind Velocity', name='wind'))
        >>> dataset.append(mogptk.LoadDataFrame(df, x_col='Date', y_col='Tidal Height', name='tidal'))

        >>> dataset = mogptk.DataSet(x, y)

        >>> dataset = mogptk.DataSet(x, [y1, y2, y3], names=['A', 'B', 'C'])
        
        >>> dataset = mogptk.DataSet([x1, x2, x3], [y1, y2, y3])

        Accessing individual channels:
        >>> dataset[0]       # first channel
        >>> dataset['wind']  # wind velocity channel
    """
    def __init__(self, *args, names=None):
        self.channels = []
        if len(args) == 2 and (isinstance(args[1], np.ndarray) or isinstance(args[1], list) and all(isinstance(item, np.ndarray) for item in args[1])):
            if names is None or isinstance(names, str):
                names = [names]

            if isinstance(args[0], np.ndarray):
                for name, y in zip(names, args[1]):
                    self.append(Data(args[0], y, name=name))
                return
            elif isinstance(args[0], list) and all(isinstance(item, np.ndarray) for item in args[0]) and isinstance(args[1], list) and len(args[0]) == len(args[1]):
                for name, x, y in zip(names, args[0], args[1]):
                    self.append(Data(x, y, name=name))
                return

        for arg in args:
            self.append(arg)

    def _format_X(self, X):
        if isinstance(X, dict):
            x_dict = X
            X = self.get_prediction()
            for name, channel_x in x_dict.items():
                X[self.get_index(name)] = channel_x
        elif isinstance(X, np.ndarray):
            X = [X] * self.get_output_dims()
        elif isinstance(X, pd.Series):
            X = [X.to_numpy()] * self.get_output_dims()
        elif not isinstance(X, list):
            raise ValueError("X must be a list, dict, numpy.ndarray, or pandas.Series")
        elif not any(isinstance(x, (list,np.ndarray)) for x in X):
            X = [X] * self.get_output_dims()
        if len(X) != self.get_output_dims():
            raise ValueError("X must be of shape (data_points,), (data_points,input_dims), or [(data_points,)] * input_dims for each channel")

        for j, channel in enumerate(self.channels):
            X[j], _ = channel._format_X(X[j])
        return X

    def __iter__(self):
        return self.channels.__iter__()

    def __len__(self):
        return len(self.channels)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.channels[self.get_names().index(key)]
        return self.channels[key]

    def __setitem__(self, key, arg):
        if isinstance(arg, Data):
            self.channels[key] = arg
        elif isinstance(arg, DataSet) and len(arg) == 1:
            self.channels[key] = arg[0]
        else:
            raise ValueError("must set a data type of Data or a DataSet with a single channel")

    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        s = ''
        for channel in self.channels:
            s += channel.__repr__() + "\n"
        return s

    def append(self, arg):
        """
        Append channel(s) to the DataSet.
        
        Args:
            arg (mogptk.data.Data, mogptk.dataset.DataSet, list, dict): Argument can be either a `DataSet` or `Data` object, a list of `Data` objects or a dictionary of `Data` objects. Each `Data` object will be added to the list of channels. In case of a dictionary, the key will set the name of the channel. If a `DataSet` is passed, its channels will be added.

        Examples:
            >>> dataset.append(mogptk.LoadFunction(lambda x: np.sin(5*x[:,0]), n=200, start=0.0, end=4.0, name='A'))
        """
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
        else:
            raise ValueError("unknown data type %s in append to DataSet" % (type(arg)))
        return self

    def copy(self):
        """
        Make a deep copy of `DataSet`.

        Returns:
            mogptk.dataset.DataSet

        Examples:
            >>> other = dataset.copy()
        """
        return copy.deepcopy(self)

    def transform(self, transformer):
        """
        Transform each channel by using one of the provided transformers, such as `TransformDetrend`, `TransformLinear`, `TransformLog`, `TransformNormalize`, `TransformStandard`, etc.

        Args:
            transformer (obj): Transformer object derived from TransformBase.

        Examples:
            >>> dataset.transform(mogptk.TransformDetrend(degree=2))        # remove polynomial trend
            >>> dataset.transform(mogptk.TransformLinear(slope=1, bias=2))  # remove linear trend
            >>> dataset.transform(mogptk.TransformLog)                      # log transform the data
            >>> dataset.transform(mogptk.TransformNormalize)                # transform to [-1,1]
            >>> dataset.transform(mogptk.TransformStandard)                 # transform to mean=0, var=1
        """
        for channel in self.channels:
            channel.transform(transformer)

    def filter(self, start, end, dim=None):
        """
        Filter the data range to be between `start` and `end` in the X axis.

        Args:
            start (float, str, list): Start of interval.
            end (float, str, list): End of interval.
            dim (int): Input dimension to apply to, if not specified applies to all input dimensions.

        Examples:
            >>> dataset.filter(3, 8)

            >>> dataset.filter('2016-01-15', '2016-06-15')
        """
        for channel in self.channels:
            channel.filter(start, end, dim=dim)

    def aggregate(self, duration, f=np.mean):
        """
        Aggregate the data by duration and apply a function to obtain a reduced dataset.

        For example, group daily data by week and take the mean. The duration can be set as a number which defined the intervals on the X axis, or by a string written in the duration format in case the X axis has data type `numpy.datetime64`. The duration format uses: Y=year, M=month, W=week, D=day, h=hour, m=minute, and s=second. For example, 3W1D means three weeks and one day, ie. 22 days, or 6M to mean six months.

        Args:
            duration (float, str): Duration along the X axis or as a string in the duration format.
            f (function): Function to use to reduce data.

        Examples:
            >>> dataset.aggregate(5)

            >>> dataset.aggregate('2W', f=np.sum)
        """
        for channel in self.channels:
            channel.aggregate(duration, f)

    def has_test_data(self):
        """
        Returns True if observations have been removed using the `remove_*` methods.

        Returns:
            list: Boolean per channel.

        Examples:
            >>> data.has_test_data()
            True
        """
        return [channel.has_test_data() for channel in self.channels]

    def get_input_dims(self):
        """
        Return the input dimensions per channel.

        Returns:
            list: List of the number of input dimensions per channel.

        Examples:
            >>> dataset.get_input_dims()
            [2, 1]
        """
        return [channel.get_input_dims() for channel in self.channels]

    def get_output_dims(self):
        """
        Return the output dimensions of the dataset, i.e. the number of channels.

        Returns:
            int: Number of output dimensions.

        Examples:
            >>> dataset.get_output_dims()
            4
        """
        return len(self.channels)

    def get_names(self):
        """
        Return the names of the channels.

        Returns:
            list: List of channel names.

        Examples:
            >>> dataset.get_names()
            ['A', 'B', 'C']
        """
        return [channel.get_name() for i, channel in enumerate(self.channels)]

    def get(self, index):
        """
        Return Data object given a channel index or name.

        Args:
            index (int, str): Index or name of the channel.

        Returns:
            mogptk.data.Data: Channel data.

        Examples:
            >>> channel = dataset.get('A')
        """
        if isinstance(index, int):
            if index < len(self.channels):
                return self.channels[index]
        elif isinstance(index, str):
            for channel in self.channels:
                if channel.name == index:
                    return channel
        raise ValueError("channel '%d' does not exist in DataSet" % (index))
    
    def get_index(self, index):
        """
        Return channel's numeric index given its name.

        Args:
            index (int, str): Index or name of the channel.

        Returns:
            int: Channel index.

        Examples:
            >>> channel_index = dataset.get_index('A')
        """
        if isinstance(index, int):
            if index < len(self.channels):
                return index
        elif isinstance(index, str):
            for channel in self.channels:
                if channel.name == index:
                    return index
        raise ValueError("channel '%d' does not exist in DataSet" % (index))
    
    def get_data(self, transformed=False):
        """
        Returns all observations, train and test.

        Arguments:
            transformed (boolean): Return transformed data.

        Returns:
            list: X data of shape (data_points,input_dims) per channel.
            list: Y data of shape (data_points,) per channel.

        Examples:
            >>> x, y = dataset.get_data()
        """
        return [channel.get_data(transformed=transformed)[0] for channel in self.channels], [channel.get_data(transformed=transformed)[1] for channel in self.channels]
    
    def get_train_data(self, transformed=False):
        """
        Returns observations used for training.

        Arguments:
            transformed (boolean): Return transformed data.

        Returns:
            list: X data of shape (data_points,input_dims) per channel.
            list: Y data of shape (data_points,) per channel.

        Examples:
            >>> x, y = dataset.get_train_data()
        """
        return [channel.get_train_data(transformed=transformed)[0] for channel in self.channels], [channel.get_train_data(transformed=transformed)[1] for channel in self.channels]

    def get_test_data(self, transformed=False):
        """
        Returns the observations used for testing which correspond to the removed points.

        Arguments:
            transformed (boolean): Return transformed data.

        Returns:
            list: X data of shape (data_points,input_dims) per channel.
            list: Y data of shape (data_points,) per channel.

        Examples:
            >>> x, y = dataset.get_test_data()
        """
        return [channel.get_test_data(transformed=transformed)[0] for channel in self.channels], [channel.get_test_data(transformed=transformed)[1] for channel in self.channels]

    def get_prediction_data(self):
        """
        Returns the prediction X range for all channels.

        Returns:
            list: X prediction of shape (data_points,input_dims) per channel.

        Examples:
            >>> x = dataset.get_prediction_data()
        """
        x = []
        for channel in self.channels:
            x.append(channel.get_prediction_data())
        return x

    def set_prediction_data(self, X):
        """
        Set the prediction range directly for saved predictions per channel. This will clear old predictions.

        Args:
            X (list, dict): Array of shape (data_points,), (data_points,input_dims), or [(data_points,)] * input_dims per channel with prediction X values. If a dictionary is passed, the index is the channel index or name.

        Examples:
            >>> dataset.set_prediction_data([[5.0, 5.5, 6.0, 6.5, 7.0], [0.1, 0.2, 0.3]])
            >>> dataset.set_prediction_data({'A': [5.0, 5.5, 6.0, 6.5, 7.0], 'B': [0.1, 0.2, 0.3]})
        """
        if isinstance(X, list):
            if len(X) != len(self.channels):
                raise ValueError("prediction x expected to be a list of shape (output_dims,n)")

            for i, channel in enumerate(self.channels):
                channel.set_prediction_data(X[i])
        elif isinstance(X, dict):
            for name in X:
                self.get(name).set_prediction_data(X[name])
        else:
            for i, channel in enumerate(self.channels):
                channel.set_prediction_data(X)

    def set_prediction_range(self, start, end, n=None, step=None):
        """
        Set the prediction range per channel. Inputs should be lists of shape (input_dims,) for each channel or dicts where the keys are the channel indices.

        Args:
            start (list, dict): Start values for prediction range per channel.
            end (list, dict): End values for prediction range per channel.
            n (list, dict): Number of points for prediction range per channel.
            step (list, dict): Step size for prediction range per channel.

        Examples:
            >>> dataset.set_prediction_range([2, 3], [5, 6], [4, None], [None, 0.5])
            >>> dataset.set_prediction_range(0.0, 5.0, n=200) # the same for each channel
        """
        if not isinstance(start, (list, dict)):
            start = [start] * self.get_output_dims()
        elif isinstance(start, dict):
            start = [start[name] for name in self.get_names()]
        if not isinstance(end, (list, dict)):
            end = [end] * self.get_output_dims()
        elif isinstance(end, dict):
            end = [end[name] for name in self.get_names()]
        if n is None:
            n = [None] * self.get_output_dims()
        elif not isinstance(n, (list, dict)):
            n = [n] * self.get_output_dims()
        elif isinstance(n, dict):
            n = [n[name] for name in self.get_names()]
        if step is None:
            step = [None] * self.get_output_dims()
        elif not isinstance(step, (list, dict)):
            step = [step] * self.get_output_dims()
        elif isinstance(step, dict):
            step = [step[name] for name in self.get_names()]

        if len(start) != len(self.channels) or len(end) != len(self.channels) or len(n) != len(self.channels) or len(step) != len(self.channels):
            raise ValueError("start, end, n, and/or step must be lists of shape (output_dims,n)")

        for i, channel in enumerate(self.channels):
            channel.set_prediction_range(start[i], end[i], n[i], step[i])
    
    def get_nyquist_estimation(self):
        """
        Estimate the Nyquist frequency by taking 0.5/(minimum distance of points) per channel.

        Returns:
            list: Nyquist frequency array of shape (input_dims) per channel.

        Examples:
            >>> freqs = dataset.get_nyquist_estimation()
        """
        return [channel.get_nyquist_estimation() for channel in self.channels]
    
    def get_ls_estimation(self, Q=1, n=10000):
        """
        Peak estimation of the spectrum using Lomb-Scargle per channel.

        Args:
            Q (int): Number of peaks to find.
            n (int): Number of points of the grid to evaluate frequencies.

        Returns:
            list: Amplitude array of shape (Q,input_dims) per channel.
            list: Frequency array of shape (Q,input_dims) per channel.
            list: Variance array of shape (Q,input_dims) per channel.

        Examples:
            >>> amplitudes, means, variances = dataset.get_lombscargle_estimation()
        """
        amplitudes = []
        means = []
        variances = []
        for channel in self.channels:
            channel_amplitudes, channel_means, channel_variances = channel.get_ls_estimation(Q, n)
            amplitudes.append(channel_amplitudes)
            means.append(channel_means)
            variances.append(channel_variances)
        return amplitudes, means, variances
    
    def get_bnse_estimation(self, Q=1, n=1000, iters=200):
        """
        Peak estimation of the spectrum using BNSE (Bayesian Non-parametric Spectral Estimation) per channel.

        Args:
            Q (int): Number of peaks to find.
            n (int): Number of points of the grid to evaluate frequencies.
            iters (str): Maximum iterations.

        Returns:
            list: Amplitude array of shape (Q,input_dims) per channel.
            list: Frequency array of shape (Q,input_dims) per channel.
            list: Variance array of shape (Q,input_dims) per channel.

        Examples:
            >>> amplitudes, means, variances = dataset.get_bnse_estimation()
        """
        amplitudes = []
        means = []
        variances = []
        for channel in self.channels:
            channel_amplitudes, channel_means, channel_variances = channel.get_bnse_estimation(Q, n, iters=iters)
            amplitudes.append(channel_amplitudes)
            means.append(channel_means)
            variances.append(channel_variances)
        return amplitudes, means, variances
    
    def get_sm_estimation(self, Q=1, method='BNSE', optimizer='Adam', iters=200, params={}):
        """
        Peak estimation of the spectrum using the spectral mixture kernel per channel.

        Args:
            Q (int): Number of peaks to find.
            method (str): Method of estimation.
            optimizer (str): Optimization method.
            iters (str): Maximum iterations.
            params (object): Additional parameters for PyTorch optimizer.

        Returns:
            list: Amplitude array of shape (Q,input_dims) per channel.
            list: Frequency array of shape (Q,input_dims) per channel.
            list: Variance array of shape (Q,input_dims) per channel.

        Examples:
            >>> amplitudes, means, variances = dataset.get_sm_estimation()
        """
        amplitudes = []
        means = []
        variances = []
        for channel in self.channels:
            channel_amplitudes, channel_means, channel_variances = channel.get_sm_estimation(Q, method, optimizer, iters, params)
            amplitudes.append(channel_amplitudes)
            means.append(channel_means)
            variances.append(channel_variances)
        return amplitudes, means, variances

    def plot(self, pred=None, title=None, figsize=None, legend=True, transformed=False):
        """
        Plot the data including removed observations, latent function, and predictions for each channel.

        Args:
            pred (str): Specify model name to draw.
            title (str): Set the title of the plot.
            figsize (tuple): Set the figure size.
            legend (boolean): Disable legend.
            transformed (boolean): Display transformed Y data as used for training.

        Returns:
            matplotlib.figure.Figure: The figure.
            list of matplotlib.axes.Axes: List of axes.

        Examples:
            >>> fig, axes = dataset.plot(title='Title')
        """
        if figsize is None:
            figsize = (12,4*len(self.channels))

        h = figsize[1]
        fig, axes = plt.subplots(self.get_output_dims(), 1, figsize=figsize, squeeze=False, constrained_layout=True)

        legends = {}
        for channel in range(self.get_output_dims()):
            self.channels[channel].plot(pred=pred, ax=axes[channel,0], transformed=transformed)
            l = axes[channel,0].get_legend()
            for text, handle in zip(l.texts, l.legendHandles):
                if text.get_text() == "Observations":
                    handle = plt.Line2D([0], [0], ls='', color='r', marker='.', ms=10, label='Observations')
                legends[text.get_text()] = handle
            l.remove()

        legend_rows = (len(legends)-1)/5 + 1
        if title is not None:
            fig.suptitle(title, y=(h+0.2+0.4*legend_rows)/h, fontsize=18)

        if legend:
            fig.legend(handles=legends.values(), ncol=5)
        return fig, axes

    def plot_spectrum(self, title=None, method='ls', per=None, maxfreq=None, figsize=None, log=False, transformed=True, n=1001):
        """
        Plot the spectrum for each channel.

        Args:
            title (str): Set the title of the plot.
            method (list, str): Set the method to get the spectrum such as LS or BNSE.
            per (list, str): Set the scale of the X axis depending on the formatter used, eg. per=5, per='day', or per='3D'.
            maxfreq (list, float): Maximum frequency to plot, otherwise the Nyquist frequency is used.
            figsize (tuple): Set the figure size.
            log (boolean): Show X and Y axis in log-scale.
            transformed (boolean): Display transformed Y data as used for training.
            n (int): Number of points used for periodogram.

        Returns:
            matplotlib.figure.Figure: The figure.
            list of matplotlib.axes.Axes: List of axes.

        Examples:
            >>> fig, axes = dataset.plot_spectrum(title='Title', method='bnse')
        """
        if not isinstance(method, list):
            method = [method] * len(self.channels)
        if not isinstance(per, list):
            per = [per] * len(self.channels)
        if not isinstance(maxfreq, list):
            maxfreq = [maxfreq] * len(self.channels)

        if figsize is None:
            figsize = (12,4*len(self.channels))

        fig, axes = plt.subplots(self.get_output_dims(), 1, figsize=figsize, squeeze=False, constrained_layout=True)
        if title != None:
            fig.suptitle(title, fontsize=18)

        for channel in range(self.get_output_dims()):
            self.channels[channel].plot_spectrum(method=method[channel], ax=axes[channel,0], per=per[channel], maxfreq=maxfreq[channel], log=log, transformed=transformed, n=n)
        return fig, axes
