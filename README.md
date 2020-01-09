# Multi-Output Gaussian Process Toolkit

[Documentation](https://games-uchile.github.io/MultiOutputGP-Toolkit/)

The Multi-Output Gaussian Process Toolkit is a Python toolkit for training and interpreting Gaussian process models with multiple data channels. It builds upon [GPflow](https://www.gpflow.org/) and [TensorFlow](https://www.tensorflow.org/) to provide an easy way to train multi-output models effectively and interpret their results. The main authors are Taco de Wolff, Alejandro Cuevas, and Felipe Tobar as part of the Center for Mathematical Modelling at the University of Chile.

## Installation guide
Make sure you have Python 3.7 and `pip` installed, and run the following command:

```
pip install git+https://github.com/GAMES-UChile/MultiOutputGP-Toolkit
```

This will automatically install the necessary dependencies (such as GPflow2 and TensorFlow2).

## Documentation
See **[Documentation](https://games-uchile.github.io/MultiOutputGP-Toolkit/)** for documentation of our toolkit, including usage and examples.

## Introduction
This repository provides a toolkit to perform multi-output GP regression with kernels that are designed to utilize correlation information among channels in order to better model signals. The toolkit is mainly targeted to time-series, and includes plotting functions for the case of single input with multiple outputs (time series with several channels).

The main kernel corresponds to Multi Output Spectral Mixture Kernel, which correlates every pair of data points (irrespective of their channel of origin) to model the signals. This kernel is specified in detail in the following publication: G. Parra, F. Tobar, Spectral Mixture Kernels for Multi-Output Gaussian Processes, Advances in Neural Information Processing Systems, 2017. Proceedings link: http://papers.nips.cc/paper/7245-spectral-mixture-kernels-for-multi-output-gaussian-processes

The kernel learns the cross-channel correlations of the data, so it is particularly well-suited for the task of signal reconstruction in the event of sporadic data loss. All other included kernels can be derived from the Multi Output Spectral Mixture kernel by restricting some parameters or applying some transformations.

One of the main advantages of the present toolkit is the GPU support, which enables the user to train models through TensorFlow, speeding computations significantly. It also includes sparse-variational GP regression functionality, to decrease computation time even further.

## Getting Started
Once installed, you can run the following example to see how the toolkit works. Using a variety of utility function you can load and filter/transform/aggregate your data before feeding it to the available models. Each model has additional functionality for parameter estimation or interpretation.

```
import numpy as np
import mogptk

# Create artificial dataset of three channels composed of sinusses
data = mogptk.DataSet()
data.append(mogptk.LoadFunction(lambda x: np.sin(5*x[:,0]), n=200, start=0.0, end=4.0, name='A'))
data.append(mogptk.LoadFunction(lambda x: np.sin(6*x[:,0])+2, n=200, start=0.0, end=4.0, var=0.03, name='B'))
data.append(mogptk.LoadFunction(lambda x: np.sin(6*x[:,0])+2 - np.sin(4*x[:,0]), n=20, start=0.0, end=4.0, var=0.03, name='C'))

# Remove a range of data from the first channel in order to imputate missing data
data['A'].remove_random_ranges(3, 0.5)

# Set a prediction range for all channels in order to predict data
data.set_pred_range(0.0, 5.0, n=200)


# Set up the MOSM kernel with three components
mosm = mogptk.MOSM(data, Q=3)
mosm.print_params()

# Do an initial estimation of parameters by training independent SM kernels for each channel
mosm.estimate_params(method='SM')
mosm.print_params()

# Train our MOSM model
mosm.train()
mosm.print_params()

# Predict data given the prediction data range previously defined
mosm.predict()

# Plot the results
data.plot()
```

## Glossary
- GP - Gaussian process, see [Gaussian Processes for Machine Learning](http://www.gaussianprocess.org/gpml/) by C.E. Rasmussen and C.K.I. Williams.
- M - the number of channels (i.e. output dimensions)
- N - the number of data points
- Q - the number of components to use for a kernel. Each component is added additively to the kernel set using the GPflow kernel addition operator
- MOSM - Multi-output spectral mixture kernel, see [Spectral Mixture Kernels for Multi-Output Gaussian Processes](https://arxiv.org/abs/1709.01298) by G. Parra and F. Tobar.
- CSM - Cross spectral mixture kernel, see [GP Kernels for Cross-Spectrum Analysis](https://papers.nips.cc/paper/5966-gp-kernels-for-cross-spectrum-analysis) by K.R. Ulrich et al.
- CONV - Convolution Gaussian kernel, see [Sparse Convolved Multiple Output Gaussian Processes](https://arxiv.org/abs/0911.5107) by M.A. Álvarez and N.D. Lawrence.
- SM-LMC - Spectral mixture linear model of coregionalization kernel, see [Gaussian Process Kernels for Pattern Discovery and Extrapolation](https://arxiv.org/abs/1302.4245) by A.G. Wilson and R.P. Adams and the book "Geostatistics for Natural Resource Evaluation" by P. Goovaerts.

## Main authors
- Taco de Wolff
- Alejandro Cuevas
- Felipe Tobar

## Contributing
We accept and encourage contributions to the toolkit in the form of pull requests (PRs), bug reports and discussions (GitHub issues). Please see TODO for more information.

## Citing
TODO

## License
Released under the [MIT license](LICENSE).
