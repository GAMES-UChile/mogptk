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

## Implementation
Implemented models:

- Exact
- Snelson (E. Snelson, Z. Ghahramani, "Sparse Gaussian Processes using Pseudo-inputs", 2005)
- OpperArchambeau (M. Opper, C. Archambeau, "The Variational Gaussian Approximation Revisited", 2009)
- Titsias (Titsias, "Variational learning of induced variables in sparse Gaussian processes", 2009)
- Hensman (J. Hensman, et al., "Scalable Variational Gaussian Process Classification", 2015)

Implemented likelihoods:

- Gaussian
- Student-T
- Exponential
- Laplace
- Bernoulli
- Beta
- Gamma
- Poisson
- Weibull
- Log-Logistic
- Log-Gaussian
- Chi
- Chi-Squared

## Tutorials

**[00 - Quick Start](https://games-uchile.github.io/mogptk/examples.html?q=00_Quick_Start)**: Short notebook showing the basic use of the toolkit.

**[01 - Data Loading](https://games-uchile.github.io/mogptk/examples.html?q=01_Data_Loading)**: Functionality to load CSVs and DataFrames while using formatters for dates.

**[02 - Data Preparation](https://games-uchile.github.io/mogptk/examples.html?q=02_Data_Preparation)**: Handle data, removing observations to simulate sensor failure and apply tranformations to the data.

**[03 - Parameter Initialization](https://games-uchile.github.io/mogptk/examples.html?q=03_Parameter_Initialization)**: Parameter initialization using different methods, for single output regression using spectral mixture kernel and multioutput case using MOSM kernel.

**[04 - Model Training](https://games-uchile.github.io/mogptk/examples.html?q=04_Model_Training)**: Training of models while keeping certain parameters fixed.

**[05 - Error Metrics](https://games-uchile.github.io/mogptk/examples.html?q=05_Error_Metrics)** Obtain different metrics in order to compare models.

**[06 - Custom Kernels and Mean Functions](https://games-uchile.github.io/mogptk/examples.html?q=06_Custom_Kernels_and_Mean_Functions)** Use or create custom kernels as well as training custom mean functions.

**[07 - Sparse Multi Input](https://games-uchile.github.io/mogptk/examples.html?q=07_Sparse_Multi_Input)** Use 8 input dimensions to train the Abalone data set using sparse GPs.

## Examples

**[Airline passengers](https://games-uchile.github.io/mogptk/examples.html?q=example_airline_passengers)**: Regression using a single output spectral mixture on the yearly number of passengers of an airline.

**[Seasonal CO2 of Mauna-Loa](https://games-uchile.github.io/mogptk/examples.html?q=example_mauna_loa)**: Regression using a single output spectral mixture on the CO2 concentration at Mauna-Loa throughout many years.

**[Currency Exchange](https://games-uchile.github.io/mogptk/examples.html?q=example_currency_exchange)**: Model training, interpretation and comparison on a dataset of 11 currency exchange rates (against the dollar) from 2017 and 2018. These 11 channels are fitted with the MOSM, SM-LMC, CSM, and CONV kernels and their results are compared and interpreted.

**[Gold, Oil, NASDAQ, USD-index](https://games-uchile.github.io/mogptk/examples.html?q=example_gold_oil_NASDAQ_USD)**: The commodity indices for gold and oil, together with the indices for the NASDAQ and the USD against a basket of other currencies, we train multiple models to find correlations between the macro economic indicators.

**[Human Activity Recognition](https://games-uchile.github.io/mogptk/examples.html?q=example_human_activity_recognition)**: Using the Inertial Measurement Unit (IMU) of an Apple iPhone 4, the accelerometer, gyroscope and magnetometer 3D data were recorded for different activities resulting in nine channels.

**[Bramblemet tidal waves](https://games-uchile.github.io/mogptk/examples.html?q=example_bramblemet)**: Tidal wave data set of four locations in the south of England. We model the tidal wave periods of approximately 12.5 hours using different multi-output Gaussian processes.

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

## Citations
- [A.I. Cowen-Rivers, et al., SAMBA: Safe Model-Based & Active Reinforcement Learning](https://arxiv.org/abs/2006.09436)
- [O.A. Guerrero, et al., Subnational Sustainable Development: The Role of Vertical Intergovernmental Transfers in Reaching Multidimensional Goals](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3837492)
- [O.A. Guerrero, G. Castañeda, How Does Government Expenditure Impact Sustainable Development? Studying the Multidimensional Link between Budgets and Development Gaps](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3800218)
- [T.V. Vo, et al., Federated Estimation of Causal Effects from Observational Data](https://arxiv.org/abs/2106.00456)
- [Q. Lin, et al., Multi-output Gaussian process prediction for computationally expensive problems with multiple levels of fidelity](https://www.sciencedirect.com/science/article/pii/S0950705121004147?casa_token=9CDCb7EpGKUAAAAA:nn6LhsAIYn0b5o9JkRJgz4GlPY4pAYeKz-Xchf-1yxJ5czbLLw7jaBQRF3IXtcs6M1fUYkT0aEI)
- [S. Covino, et al., Detecting the periodicity of highly irregularly sampled light-curves with GPs](https://arxiv.org/abs/2203.03614)
- [Y. Jung, J. Park, Scalable Inference for Hybrid Bayesian HMM using GP Emission](https://www.tandfonline.com/doi/abs/10.1080/10618600.2021.2023021)
- [H. Liu, et al., Scalable multi-task GPs with neural embedding of coregionalization](https://www.sciencedirect.com/science/article/abs/pii/S0950705122003641)
- [L.M. Rivera-Muñoz, et al., Missing Data Estimation in a Low-Cost Sensor network for Measuring Air Quality](https://link.springer.com/article/10.1007/s11270-021-05363-1)

### Used in code
- https://github.com/jdjmoon/TRF
- https://github.com/ErickRosete/Multivariate_regression
- https://github.com/clara-risk/fire_weather_interpolate
- https://github.com/becre2021/multichannels-corrnp
- https://github.com/ArthurLeroy/MAGMAclust
- https://github.com/nicdel-git/master_thesis


## License
Released under the [MIT license](LICENSE).
