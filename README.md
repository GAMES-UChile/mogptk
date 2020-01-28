# Multi-Output Gaussian Process Toolkit

**[API Documentation](https://games-uchile.github.io/MultiOutputGP-Toolkit/) - [Tutorials & Examples](https://github.com/GAMES-UChile/MultiOutputGP-Toolkit#tutorials)**

The Multi-Output Gaussian Process Toolkit is a Python toolkit for training and interpreting Gaussian process models with multiple data channels. It builds upon [GPflow](https://www.gpflow.org/) and [TensorFlow](https://www.tensorflow.org/) to provide an easy way to train multi-output models effectively and interpret their results. The main authors are Taco de Wolff, Alejandro Cuevas, and Felipe Tobar as part of the Center for Mathematical Modelling at the University of Chile.

## Installation
Make sure you have at least Python 3.6 and run the following command using pip:

```
pip install mogptk
```

This will automatically install the necessary dependencies such as GPflow2 and TensorFlow2. See [Tutorials & Examples](https://github.com/GAMES-UChile/MultiOutputGP-Toolkit#tutorials) to get started.

## Documentation
See the [API documentation](https://games-uchile.github.io/MultiOutputGP-Toolkit/) for documentation of our toolkit, including usage and examples of functions and classes.

## Introduction
This repository provides a toolkit to perform multi-output GP regression with kernels that are designed to utilize correlation information among channels in order to better model signals. The toolkit is mainly targeted to time-series, and includes plotting functions for the case of single input with multiple outputs (time series with several channels).

The main kernel corresponds to Multi Output Spectral Mixture Kernel, which correlates every pair of data points (irrespective of their channel of origin) to model the signals. This kernel is specified in detail in the following publication: G. Parra, F. Tobar, Spectral Mixture Kernels for Multi-Output Gaussian Processes, Advances in Neural Information Processing Systems, 2017. Proceedings link: http://papers.nips.cc/paper/7245-spectral-mixture-kernels-for-multi-output-gaussian-processes

The kernel learns the cross-channel correlations of the data, so it is particularly well-suited for the task of signal reconstruction in the event of sporadic data loss. All other included kernels can be derived from the Multi Output Spectral Mixture kernel by restricting some parameters or applying some transformations.

One of the main advantages of the present toolkit is the GPU support, which enables the user to train models through TensorFlow, speeding computations significantly. It also includes sparse-variational GP regression functionality, to decrease computation time even further.

## Tutorials

**[00 - Quick Start](https://github.com/GAMES-UChile/MultiOutputGP-Toolkit/blob/master/examples/00_Quick_Start.ipynb)**: Short notebook showcasing basic use of the toolkit.

**[01 - Data Loading](https://github.com/GAMES-UChile/MultiOutputGP-Toolkit/blob/master/examples/01_Data_Loading.ipynb)**: CSV & DataFrame

**[02 - Data Preparation](https://github.com/GAMES-UChile/MultiOutputGP-Toolkit/blob/master/examples/02_Data_Preparation.ipynb)**: Use different transformations on a financial dataset composed of four channels (Gold, Oil, NASDAQ, and the US dollar index). [Incomplete]

**[03 - Parameter Estimation](https://github.com/GAMES-UChile/MultiOutputGP-Toolkit/blob/master/examples/03_Parameter_Estimation.ipynb)**: Parameter initialization using different methods, for single output regression using spectral mixture kernel and multioutput case using MOSM kernel. [Incomplete]

**[04 - Model Training](https://github.com/GAMES-UChile/MultiOutputGP-Toolkit/blob/master/examples/04_Model_Training.ipynb)**

**[05 - Error Metrics](https://github.com/GAMES-UChile/MultiOutputGP-Toolkit/blob/master/examples/05_Error_Metrics.ipynb)** Obtain different metrics to compare models using climate dataset.

## Examples

**[Currency Exchange](https://github.com/GAMES-UChile/MultiOutputGP-Toolkit/blob/master/examples/currency_exchange_experiment.ipynb)**: Model training, interpretation and comparison on a dataset of 11 currency exchanges in 2017 and 2018 with respect to the US dollar. These 11 channels are fitted with the MOSM, SM-LMC, CSM and CONV kernel and their results are compared. We also interpret the results for some models and show how much some channels correlate.

**[Gold, Oil, NASDAQ, USD-index](https://github.com/GAMES-UChile/MultiOutputGP-Toolkit/blob/master/examples/example_GONU.ipynb)**

**[Human Activity Recognition](https://github.com/GAMES-UChile/MultiOutputGP-Toolkit/blob/master/examples/example_HAR.ipynb)**

<!--
## Getting Started
Once installed, you can follow the following example to see how the toolkit works. Using a variety of utility function you can load and filter/transform/aggregate your data before feeding it to the available models. Each model has additional functionality for parameter estimation or interpretation.

This is a quick example showing how this library can be used. First we import the library as follows:

```python
import mogptk
```

Next we load our data (`x`, and `y`) into a `Data` class. Your `x` data should be a `list` or `numpy.ndarray` of `n` data points. When you have more than one input dimension, the shape of `x` should look like `(n,input_dims)`. When `x` is a dictionary, each item should be a `list` or `numpy.ndarray`, and the keys should be referenced by passing a `x_labels` which is a `list` or strings. Your `y` data should be a `list` or `numpy.ndarray` of `n` data points.

```python
data = mogptk.Data(x, y, name='Dataset')
```

If you have multiple output channels it is convenient to use the `DataSet` class which can hold multiple `Data` objects, one for each channel. Additionally, there is functionality to create a `Data` object from a CSV file, a `pandas.DataFrame` or using a Python function, for example:

```python
data = mogptk.DataSet()
data.append(mogptk.LoadFunction(lambda x: np.sin(5*x[:,0]), n=200, start=0.0, end=4.0, name='Function'))
data.append(mogptk.LoadCSV('data.csv', 'time', 'measurement', name='CSV'))
data.append(mogptk.LoadDataFrame(df, 'x', 'y', name='DataFrame'))
```

Next we can transform our data, remove data points (e.g. to simulate sensor failure), and set our prediction range:

```python
data['CSV'].transform(mogptk.TransformLog)
data['DataFrame'].remove_randomly(pct=0.40)
data['Function'].set_pred_range(0.0, 5.0, n=200)
```

With our data set, we can instantiate a model (e.g. `MOSM`, `CSM`, `CONV`, `SM_LMC`) and set its parameters. All models accept a `Q` parameter which is the number of (Gaussian) mixtures to use. Some models also have an `Rq` parameter, see the cited references above for an exact description.

```python
mosm = mogptk.MOSM(data, Q=3)
```

Given our model, we first estimate the parameters using a per-channel Spectral Mixture kernel or using Bayesian Nonparametric Spectral Estimation to improve the speed and likelihood of convergence when training:

```python
mosm.estimate_params()
```

Next we train our model and do a prediction using the data range previously defined:

```python
mosm.train()
mosm.predict()
```

Finally we can plot our data and prediction:

```python
data.plot()
```
-->

## Authors
- Taco de Wolff
- Alejandro Cuevas
- Felipe Tobar

## Contributing
We accept and encourage contributions to the toolkit in the form of pull requests (PRs), bug reports and discussions (GitHub issues). TODO

## Citing
TODO

## License
Released under the [MIT license](LICENSE).
