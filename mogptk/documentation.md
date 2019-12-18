MOGPTK is a Python toolkit for multi output Gaussian processes. It contains data handling classes and different multi output models to facilitate the training and predicting of multi channel data sets. It provides a complete toolkit to improve the training of MOGPs, where it allows for easy reformatting of data (date/time to numbers), detrending or transformation of the data (such as taking the natural logarithm), functions that aid in adding gaps to the data or specifying which range should be predicted, and more.

It contains the implementation of four MOGP models:

- Multi Output Spectral Mixture (MOSM) as proposed by [1]
- Cross Spectral Mixture (CSM) as proposed by [2]
- Convolutional Gaussian (CONV) as proposed by [3]
- Spectral Mixture - Linear Model of Coregionalization (SM-LMC) as proposed by combining SM [4] and LMC [5]

Each model has specialized methods for parameter estimation to improve training. For example, the MOSM model can estimate its parameters using the training of spectral mixture kernels to each of the channels individually. It can also estimate its parameters using Bayesian Nonparametric Spectral Estimation (BNSE) [6]. After parameter estimation, the training of the model will converge more rapidly and there is a higher chance it will converge. Additionally, some models allow the interpretation of the parameters by investigating the cross-correlation parameters between the channels to better interpret what the learned parameters actually mean on your data.

- [1] G. Parra and F. Tobar, "Spectral Mixture Kernels for Multi-Output Gaussian Processes", Advances in Neural Information Processing Systems 31, 2017
- [2] K.R. Ulrich et al, "GP Kernels for Cross-Spectrum Analysis", Advances in Neural Information Processing Systems 28, 2015
- [3] M.A. Álvarez and N.D. Lawrence, "Sparse Convolved Multiple Output Gaussian Processes", Advances in Neural Information Processing Systems 21, 2009
- [4] A.G. Wilson and R.P. Adams, "Gaussian Process Kernels for Pattern Discovery and Extrapolation", International Conference on Machine Learning 30, 2013
- [5] P. Goovaerts, "Geostatistics for Natural Resource Evaluation", Oxford University Press, 1997
- [6] F. Tobar, "Bayesian Nonparametric Spectral Estimation", Advances in Neural Information Processing Systems 32, 2018

## Installation

## Quick Example
This is a quick example showing how this library can be used. First we import the library as follows:

    >>> import mogptk

Next we load our data (`x`, and `y`) into a `Data` class. You `x` data should be a `list` or `numpy.ndarray` of `n` data points. When you have more than one input dimension, the shape of `x` should look like `(n,input_dims)`. You `y` data should be a `list` or `numpy.ndarray` of `n` data points.

    >>> data = mogptk.Data(x, y, name='Dataset')

If you have multiple output channels, it is convenient to use the `DataSet` class which can hold multiple `Data` objects, one for each channel. Additionally, there is functionality to create a `Data` object from a CSV file, a `pandas.DataFrame` or using a Python function, for example:

    >>> data = mogptk.DataSet()
    >>> data.append(mogptk.LoadFunction(lambda x: np.sin(5*x[:,0]), n=200, start=0.0, end=4.0, name='Function'))
    >>> data.append(mogptk.LoadCSV('data.csv', 'time', 'measurement', name='CSV'))
    >>> data.append(mogptk.LoadDataFrame(df, 'x', 'y', name='DataFrame'))

Next we can transform our data, remove data points (e.g. to simulate sensor failure), and set our prediction range:

    >>> data['CSV'].transform(mogptk.TransformLog)
    >>> data['DataFrame'].remove_randomly(pct=0.40)
    >>> data['Function'].set_pred_range(0.0, 5.0, n=200)

With our data set, we can instantiate a model (e.g. `MOSM`, `CSM`, `CONV`, `SM_LMC`) and set its parameters. All models accept a `Q` parameter, which is the number of (Gaussian) mixtures to use. Some models also have an `Rq` parameter, see the cited references above for an exact description.

    >>> mosm = mogptk.MOSM(data, Q=3)

Given our model, we first estimate the parameters using a per-channel Spectral Mixture kernel or using Bayesian Nonparametric Spectral Estimation to improve the speed and likelihood of convergence when training:

    >>> mosm.estimate_params()

Next we train our model and do a prediction using the data range previously defined:

    >>> mosm.train()
    >>> mosm.predict()

Finally we can plot our data and prediction, get information about the model, etc.:

    >>> mosm.info()
    >>> data.plot()

## Models
### MOSM
### CSM
### CONV
### \*-LMC
### \*-IGP

## Data handling
### Formats
The format classes allow the independent (X axis) data to be formatted so that it can be converted into numbers. Each class implements the following functions:

- **\_parse(str) returns num**: parses a string to a number, can raise `ValueError`
- **\_parse\_duration(str) returns num**: parses a string to a number as a difference/interval, e.g. duration in seconds or distance in meters
- **\_format(num) returns str**: format a number to be displayed
- **\_scale(maxfreq=None) returns (num, str)**: returns the default duration unit of the format and its name, e.g. to plot the frequency per day or per year

#### FormatNumber
#### FormatDate
#### FormatDateTime
### Transformations
The transformation classes allow transforming the dependent data (Y axis) to be transformed. Each class implements the following functions:

- **\_data(data)**: pass the Data class in case the transformer uses that data to calculate the transformation
- **\_forward(x, y)** returns y: does a forward transformation where x has shape (n,input\_dims) and y has shape (n,)
- **\_backward(x, y)** returns y: does a backward transformation (invert) where x has shape (n,input\_dims) and y has shape (n,)

#### TransformDetrend
#### TransformNormalize
#### TransformLog
