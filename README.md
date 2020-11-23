# Multi-Output Gaussian Process Toolkit

**[Paper](https://doi.org/10.1016/j.neucom.2020.09.085) - [API Documentation](https://games-uchile.github.io/mogptk/) - [Tutorials & Examples](https://github.com/GAMES-UChile/mogptk#tutorials)**

The Multi-Output Gaussian Process Toolkit is a Python toolkit for training and interpreting Gaussian process models with multiple data channels. It builds upon [PyTorch](https://pytorch.org/) to provide an easy way to train multi-output models effectively on CPUs and GPUs. The main authors are Taco de Wolff, Alejandro Cuevas, and Felipe Tobar as part of the Center for Mathematical Modelling at the University of Chile.

## Installation
With [Anaconda](https://www.anaconda.com/distribution/) installed on your system, open a command prompt and create a virtual environment:

```
conda create -n myenv python=3.7
conda activate myenv
```

where `myenv` is the name of your environment, and where the version of Python could be 3.6 or above. Next we will install this toolkit and automatically install the necessary dependencies such as PyTorch.

```
pip install mogptk
```

In order to upgrade to a new version of MOGPTK or any of its dependencies, use `--upgrade` as follows:

```
pip install --upgrade mogptk
```

For developers of the library or for users who need the latest changes, we recommend cloning the git `master` or `develop` branch and to use the following command inside the repository folder:

```
pip install --upgrade -e .
```

See [Tutorials & Examples](https://github.com/GAMES-UChile/mogptk#tutorials) to get started.

## Introduction
This repository provides a toolkit to perform multi-output GP regression with kernels that are designed to utilize correlation information among channels in order to better model signals. The toolkit is mainly targeted to time-series, and includes plotting functions for the case of single input with multiple outputs (time series with several channels).

The main kernel corresponds to Multi Output Spectral Mixture Kernel, which correlates every pair of data points (irrespective of their channel of origin) to model the signals. This kernel is specified in detail in the following publication: G. Parra, F. Tobar, Spectral Mixture Kernels for Multi-Output Gaussian Processes, Advances in Neural Information Processing Systems, 2017. Proceedings link: https://papers.nips.cc/paper/7245-spectral-mixture-kernels-for-multi-output-gaussian-processes

The kernel learns the cross-channel correlations of the data, so it is particularly well-suited for the task of signal reconstruction in the event of sporadic data loss. All other included kernels can be derived from the Multi Output Spectral Mixture kernel by restricting some parameters or applying some transformations.

One of the main advantages of the present toolkit is the GPU support, which enables the user to train models through PyTorch, speeding computations significantly. It also includes sparse-variational GP regression functionality to decrease computation time even further.

See [MOGPTK: The Multi-Output Gaussian Process Toolkit](https://doi.org/10.1016/j.neucom.2020.09.085) for our publication in Neurocomputing.

## Tutorials

**[00 - Quick Start](https://github.com/GAMES-UChile/mogptk/blob/master/examples/00_Quick_Start.ipynb)**: Short notebook showing the basic use of the toolkit.

**[01 - Data Loading](https://github.com/GAMES-UChile/mogptk/blob/master/examples/01_Data_Loading.ipynb)**: Functionality to load CSVs and DataFrames while using formatters for dates.

**[02 - Data Preparation](https://github.com/GAMES-UChile/mogptk/blob/master/examples/02_Data_Preparation.ipynb)**: Handle data, removing observations to simulate sensor failure and apply tranformations to the data.

**[03 - Parameter Initialization](https://github.com/GAMES-UChile/mogptk/blob/master/examples/03_Parameter_Initialization.ipynb)**: Parameter initialization using different methods, for single output regression using spectral mixture kernel and multioutput case using MOSM kernel.

**[04 - Model Training](https://github.com/GAMES-UChile/mogptk/blob/master/examples/04_Model_Training.ipynb)**: Training of models while keeping certain parameters fixed.

**[05 - Error Metrics](https://github.com/GAMES-UChile/mogptk/blob/master/examples/05_Error_Metrics.ipynb)** Obtain different metrics in order to compare models.

**[06 - Custom Kernels and Mean Functions](https://github.com/GAMES-UChile/mogptk/blob/master/examples/06_Custom_Kernels_and_Mean_Functions.ipynb)** Use or create custom kernels as well as training custom mean functions.

## Examples

**[Airline passangers](https://github.com/GAMES-UChile/mogptk/blob/master/examples/example_airline_passengers.ipynb)**: Regression using a single output spectral mixture on the yearly number of passengers of an airline.

**[Seasonal CO2 of Mauna-Loa](https://github.com/GAMES-UChile/mogptk/blob/master/examples/example_mauna_loa.ipynb)**: Regression using a single output spectral mixture on the CO2 concentration at Mauna-Loa throughout many years.

**[Currency Exchange](https://github.com/GAMES-UChile/mogptk/blob/master/examples/example_currency_exchange.ipynb)**: Model training, interpretation and comparison on a dataset of 11 currency exchange rates (against the dollar) from 2017 and 2018. These 11 channels are fitted with the MOSM, SM-LMC, CSM, and CONV kernels and their results are compared and interpreted.

**[Gold, Oil, NASDAQ, USD-index](https://github.com/GAMES-UChile/mogptk/blob/master/examples/example_gold_oil_NASDAQ_USD.ipynb)**: The commodity indices for gold and oil, together with the indices for the NASDAQ and the USD against a basket of other currencies, we train multiple models to find correlations between the macro economic indicators.

**[Human Activity Recognition](https://github.com/GAMES-UChile/mogptk/blob/master/examples/example_human_activity_recognition.ipynb)**: Using the Inertial Measurement Unit (IMU) of an Apple iPhone 4, the accelerometer, gyroscope and magnetometer 3D data were recorded for different activities resulting in nine channels.

**[Bramblemet tidal waves](https://github.com/GAMES-UChile/mogptk/blob/master/examples/example_bramblemet.ipynb)**: Tidal wave data set of four locations in the south of England. We model the tidal wave periods of approximately 12.5 hours using different multi-output Gaussian processes.

## Documentation
See the [API documentation](https://games-uchile.github.io/mogptk/) for documentation of our toolkit, including usage and examples of functions and classes.

## Authors
- Taco de Wolff
- Alejandro Cuevas
- Felipe Tobar

## Users
This is a list of users of this toolbox, feel free to add your project!

## Contributing
We accept and encourage contributions to the toolkit in the form of pull requests (PRs), bug reports and discussions (GitHub issues). It is adviced to start an open discussion before proposing large PRs. For small PRs we suggest that they address only one issue or add one new feature. All PRs should keep documentation and notebooks up to date.

## Citing
Please use our publication at arXiv to cite our toolkit: [MOGPTK: The Multi-Output Gaussian Process Toolkit](https://doi.org/10.1016/j.neucom.2020.09.085). We recommend the following BibTeX entry:

```
@article{mogptk,
    author = {T. {de Wolff} and A. {Cuevas} and F. {Tobar}},
    title = {{MOGPTK: The Multi-Output Gaussian Process Toolkit}},
    journal = "Neurocomputing",
    year = "2020",
    issn = "0925-2312",
    doi = "https://doi.org/10.1016/j.neucom.2020.09.085",
    url = "https://github.com/GAMES-UChile/mogptk"
}
```

## License
Released under the [MIT license](LICENSE).
