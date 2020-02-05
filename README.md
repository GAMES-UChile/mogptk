# Multi-Output Gaussian Process Toolkit

**[API Documentation](https://games-uchile.github.io/MultiOutputGP-Toolkit/) - [Tutorials & Examples](https://github.com/GAMES-UChile/MultiOutputGP-Toolkit#tutorials)**

The Multi-Output Gaussian Process Toolkit is a Python toolkit for training and interpreting Gaussian process models with multiple data channels. It builds upon [GPflow](https://www.gpflow.org/) and [TensorFlow](https://www.tensorflow.org/) to provide an easy way to train multi-output models effectively and interpret their results. The main authors are Taco de Wolff, Alejandro Cuevas, and Felipe Tobar as part of the Center for Mathematical Modelling at the University of Chile.

## Installation
With [Anaconda](https://www.anaconda.com/distribution/) installed on your system, open a command prompt and create a virtual environment:

```
conda create -n myenv python=3.7
conda activate myenv
```

where `myenv` is the name of your environment, and where the version of Python could be 3.6 or above. In order to use TensorFlow on the GPU, the easiest way is to install TensorFlow through conda (and not pip) before we install this toolkit. If you will be using the CPU you can skip this step.

```
conda install tensorflow-gpu
```

Next we will install this toolkit and automatically install the necessary dependencies such as GPflow2 and TensorFlow2.

```
pip install mogptk
```

See [Tutorials & Examples](https://github.com/GAMES-UChile/MultiOutputGP-Toolkit#tutorials) to get started.

## Introduction
This repository provides a toolkit to perform multi-output GP regression with kernels that are designed to utilize correlation information among channels in order to better model signals. The toolkit is mainly targeted to time-series, and includes plotting functions for the case of single input with multiple outputs (time series with several channels).

The main kernel corresponds to Multi Output Spectral Mixture Kernel, which correlates every pair of data points (irrespective of their channel of origin) to model the signals. This kernel is specified in detail in the following publication: G. Parra, F. Tobar, Spectral Mixture Kernels for Multi-Output Gaussian Processes, Advances in Neural Information Processing Systems, 2017. Proceedings link: http://papers.nips.cc/paper/7245-spectral-mixture-kernels-for-multi-output-gaussian-processes

The kernel learns the cross-channel correlations of the data, so it is particularly well-suited for the task of signal reconstruction in the event of sporadic data loss. All other included kernels can be derived from the Multi Output Spectral Mixture kernel by restricting some parameters or applying some transformations.

One of the main advantages of the present toolkit is the GPU support, which enables the user to train models through TensorFlow, speeding computations significantly. It also includes sparse-variational GP regression functionality, to decrease computation time even further.

## Tutorials

**[00 - Quick Start](https://github.com/GAMES-UChile/MultiOutputGP-Toolkit/blob/master/examples/00_Quick_Start.ipynb)**: Short notebook showing the basic use of the toolkit.

**[01 - Data Loading](https://github.com/GAMES-UChile/MultiOutputGP-Toolkit/blob/master/examples/01_Data_Loading.ipynb)**: Functionality to load CSVs and DataFrames while using formatters for dates.

**[02 - Data Preparation](https://github.com/GAMES-UChile/MultiOutputGP-Toolkit/blob/master/examples/02_Data_Preparation.ipynb)**: Handle data, removing observations to simulate sensor failure and apply tranformations to the data.

**[03 - Parameter Initialization](https://github.com/GAMES-UChile/MultiOutputGP-Toolkit/blob/master/examples/03_Parameter_Initialization.ipynb)**: Parameter initialization using different methods, for single output regression using spectral mixture kernel and multioutput case using MOSM kernel.

**[04 - Model Training](https://github.com/GAMES-UChile/MultiOutputGP-Toolkit/blob/master/examples/04_Model_Training.ipynb)**: Training of models while keeping certain parameters fixed.

**[05 - Error Metrics](https://github.com/GAMES-UChile/MultiOutputGP-Toolkit/blob/master/examples/05_Error_Metrics.ipynb)** Obtain different metrics to compare models.

## Examples

**[Currency Exchange](https://github.com/GAMES-UChile/MultiOutputGP-Toolkit/blob/master/examples/example_currency_exchange.ipynb)**: Model training, interpretation and comparison on a dataset of 11 currency exchange rates (against the dollar) from 2017 and 2018. These 11 channels are fitted with the MOSM, SM-LMC, CSM, and CONV kernels and their results are compared and interpreted.

**[Gold, Oil, NASDAQ, USD-index](https://github.com/GAMES-UChile/MultiOutputGP-Toolkit/blob/master/examples/example_GONU.ipynb)**: The commodity indices for gold and oil, together with the indices for the NASDAQ and the USD against a basket of other currencies, we train multiple models to find correlations between the macro economic indicators.

**[Human Activity Recognition](https://github.com/GAMES-UChile/MultiOutputGP-Toolkit/blob/master/examples/example_HAR.ipynb)**: Using the Inertial Measurement Unit (IMU) of an Apple iPhone 4, the accelerometer, gyroscope and magnetometer 3D data were recorded for different activities resulting in nine channels.

**[Seasonal C02 and Airline passangers](https://github.com/GAMES-UChile/MultiOutputGP-Toolkit/blob/master/examples/example_single_output_spectral_mixture.ipynb)**: Regression for 2 datasets using a single output spectral mixture, first the Mauna Loa C02 concentration and the second the passangers in a airline.

## Documentation
See the [API documentation](https://games-uchile.github.io/MultiOutputGP-Toolkit/) for documentation of our toolkit, including usage and examples of functions and classes.

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
