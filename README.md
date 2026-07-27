# Multi-Output Gaussian Process Toolkit

**[Paper](https://doi.org/10.1016/j.neucom.2020.09.085) - [API Documentation](https://games-uchile.github.io/mogptk/) - [Tutorials & Examples](https://github.com/GAMES-UChile/mogptk#tutorials)  - [Code of Conduct](CODE_OF_CONDUCT.md)**

MOGPTK (Multi-Output Gaussian Process Toolkit) is an open-source Python library for interpretable probabilistic modelling of multichannel time series. Built on PyTorch, it provides Gaussian process models with a range of covariance architectures, spectral analysis tools, GPU acceleration, and visualisation utilities for scientific applications underpinned by temporal observations.

The project was initiated in 2020 at the Center for Mathematical Modelling, Universidad de Chile, and has been hosted at Imperial College London since October 2024. Development has benefited from research supported by:

- Center for Mathematical Modelling, Universidad de Chile (2020–2024)
- ANID Fondecyt research grants, Chile (2020–2024)
- Google Research Awards (2020–2024)

## Installation
With [Anaconda](https://www.anaconda.com/distribution/) installed on your system, open a command prompt and create a virtual environment:

```
conda create -n myenv python=3.14
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

For developers of the library or for users who need the latest changes, we recommend cloning the git `master` or `develop` branch and using the following command inside the repository folder:

```
pip install --upgrade -e .
```

See [Tutorials & Examples](https://github.com/GAMES-UChile/mogptk#tutorials) to get started.

## Introduction
This repository provides a toolkit to perform multi-output GP regression with kernels that are designed to utilise correlation information among channels in order to better model signals. The toolkit is mainly targeted at time series, and includes plotting functions for the case of single input with multiple outputs (time series with several channels).

The main kernel corresponds to Multi Output Spectral Mixture Kernel, which correlates every pair of data points (irrespective of their channel of origin) to model the signals. This kernel is specified in detail in the following publication: G. Parra, F. Tobar, Spectral Mixture Kernels for Multi-Output Gaussian Processes, Advances in Neural Information Processing Systems, 2017. Available [here](https://papers.nips.cc/paper/7245-spectral-mixture-kernels-for-multi-output-gaussian-processes)

The kernel learns the cross-channel correlations of the data, so it is particularly well-suited for the task of signal reconstruction in the event of sporadic data loss. All other included kernels can be derived from the Multi Output Spectral Mixture kernel by restricting some parameters or applying some transformations.

One of the main advantages of the present toolkit is the GPU support, which enables the user to train models through PyTorch, speeding computations significantly. It also includes sparse-variational GP regression functionality to decrease computation time even further.

See [MOGPTK: The Multi-Output Gaussian Process Toolkit](https://doi.org/10.1016/j.neucom.2020.09.085) for our publication in Neurocomputing.

## Features
Implemented inference models:

- Exact Gaussian process (maximum likelihood)
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

**[08 - Multi Likelihood Classification](https://games-uchile.github.io/mogptk/examples.html?q=08_Multi_Likelihood_Classification)** Use a different likelihood for each channel, one Bernoulli for classification and one StudentT's for regression.

## Examples

**[Airline passengers](https://games-uchile.github.io/mogptk/examples.html?q=example_airline_passengers)**: Regression using a single output spectral mixture on the yearly number of passengers of an airline.

**[Seasonal CO2 of Mauna-Loa](https://games-uchile.github.io/mogptk/examples.html?q=example_mauna_loa)**: Regression using a single output spectral mixture on the CO2 concentration at Mauna-Loa throughout many years.

**[Currency Exchange](https://games-uchile.github.io/mogptk/examples.html?q=example_currency_exchange)**: Model training, interpretation and comparison on a dataset of 11 currency exchange rates (against the dollar) from 2017 and 2018. These 11 channels are fitted with the MOSM, SM-LMC, CSM, and CONV kernels and their results are compared and interpreted.

**[Gold, Oil, NASDAQ, USD-index](https://games-uchile.github.io/mogptk/examples.html?q=example_gold_oil_NASDAQ_USD)**: The commodity indices for gold and oil, together with the indices for the NASDAQ and the USD against a basket of other currencies, we train multiple models to find correlations between the macro economic indicators.

**[Human Activity Recognition](https://games-uchile.github.io/mogptk/examples.html?q=example_human_activity_recognition)**: Using the Inertial Measurement Unit (IMU) of an Apple iPhone 4, the accelerometer, gyroscope and magnetometer 3D data were recorded for different activities resulting in nine channels.

**[Bramblemet tidal waves](https://games-uchile.github.io/mogptk/examples.html?q=example_bramblemet)**: Tidal wave data set of four locations in the south of England. We model the tidal wave periods of approximately 12.5 hours using different multi-output Gaussian processes.

## Documentation
See the [API documentation](https://games-uchile.github.io/mogptk/) for documentation of our toolkit, including usage and examples of functions and classes.



## Contributing
We accept and encourage contributions to the toolkit in the form of pull requests (PRs), bug reports and discussions (GitHub issues). Please consider starting an open discussion before proposing large PRs. For small PRs, we suggest that they address only one issue or add one new feature. All PRs should keep documentation and notebooks up to date. For more details, see our [Contribution Guidelines](CONTRIBUTING.md).



## Citing MOGPTK
Please refer to the publication in Neurocomputing [MOGPTK: The Multi-Output Gaussian Process Toolkit](https://doi.org/10.1016/j.neucom.2020.09.085). We recommend the following BibTeX entry:

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

## Authors
- Taco de Wolff
- Alejandro Cuevas
- Felipe Tobar

## License
Released under the [MIT license](LICENSE).



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
- [G. Caballero, et al., Synergy of Sentinel-1 and Sentinel-2 Time Series for Cloud-Free Vegetation Water Content Mapping with Multi-Output Gaussian Processes](https://www.mdpi.com/2072-4292/15/7/1822)
- [S.T. Yeh, X. Du, Optimal Tilt-Wing eVTOL Takeoff Trajectory Prediction Using Regression Generative Adversarial Networks](https://doi.org/10.3390/math12010026)
- [A. Patharkar, et al., Predictive modeling of biomedical temporal data in healthcare applications: review and future directions](https://doi.org/10.3389/fphys.2024.1386760)
- [M. Zhang, et al., A machine learning method for the prediction of ship motion trajectories in real operational conditions](https://doi.org/10.1016/j.oceaneng.2023.114905)
- [D. Polke, et al., Adaptive Learning with Gaussian Process Regression: A Comprehensive Review of Methods and Applications](https://doi.org/10.3390/make8040101)
- [O.A. Guerrero, G. Catañeda, How does government expenditure impact sustainable development? Studying the multidimensional link between budgets and development gaps](https://link.springer.com/article/10.1007/s11625-022-01095-1)
- [A.U. Hassan, M.J. Aljaafreh, Predicting UV-Vis Spectra of Benzothio/Dithiophene Polymers for Photodetectors by Machine-Learning-Assisted Computational Studies](https://doi.org/10.3390/coatings15050558)
- [J. Li, et al., Battery capacity trajectory prediction by capturing the correlation between different vehicles](https://doi.org/10.1016/j.energy.2022.125123)
- [A.U. Hassan, M.J. Aljaafreh, A Machine Learning Study to Explore the Structural Basis of Non-Conjugated Compounds for Their Optical Activity Features](https://doi.org/10.1002/adts.202500140)
- [Q. Li, M. Ludovski, Probabilistic spatiotemporal modeling of day-ahead wind power generation with input-warped Gaussian processes](https://doi.org/10.1016/j.spasta.2025.100906)
- [R.R. Griffiths, et al., Modeling the Multiwavelength Variability of Mrk 335 Using Gaussian Processes](https://iopscience.iop.org/article/10.3847/1538-4357/abfa9f/meta)
- [H.A.K. Kyhoiesh, et al., A machine learning-gaussian process screening of carbazole based donors to design efficient organic polymers for photovoltaic applications](https://doi.org/10.1016/j.jmgm.2025.109154)
- [Z. Sun, et al., PEMFC Performance Prediction Based on Degradation Mechanism and Machine Learning](https://ieeexplore.ieee.org/abstract/document/10855573)
- [J. Barahona, et al., Machine learning modeling of lung mechanics: Assessing the variability and propagation of uncertainty in respiratory-system compliance and airway resistance](https://doi.org/10.1016/j.cmpb.2023.107888)
- [E. Daş, J.W. Burdick, An Active Learning Based Robot Kinematic Calibration Framework Using Gaussian Processes](https://doi.org/10.48550/arXiv.2303.03658)
- [Y. Yang, et al., Designing strongly coupled polaritonic structures via statistical machine learning](https://doi.org/10.1073/pnas.2526690122)
- [P. Zhou, et al., Long-term prediction enhancement based on multi-output Gaussian process regression integrated with production plans for oxygen supply network](https://doi.org/10.1016/j.compchemeng.2022.107844)
- [L.M. Rivera-Muñoz, et al., Deep matrix factorization models for estimation of missing data in a low-cost sensor network to measure air quality](https://doi.org/10.1016/j.ecoinf.2022.101775)
- [X. Li, et al., Time-optimal general asymmetric S-curve profile with low residual vibration](https://doi.org/10.1016/j.ymssp.2022.109978)
- [C. Miao, Y. Wang, Multivariate Gaussian process regression for characterization of geo-data spatial variability from limited and non-co-located measurements](https://doi.org/10.1016/j.enggeo.2026.108611)
- [A. Lerow, et al., Cluster-Specific Predictions with Multi-Task Gaussian Processes](https://www.jmlr.org/papers/v24/20-1321.html)
- [T. Hoffbauer, et al., KernelMatmul: Scaling Gaussian Processes to Large Time Series](https://doi.org/10.1609/aaai.v39i16.33893)
- [J. Rohmer, et al., Improved metamodels for predicting high-dimensional outputs by accounting for the dependence structure of the latent variables: application to marine flooding](https://link.springer.com/article/10.1007/s00477-023-02426-z)
- [Y. Dai, et al., Graphical Multioutput Gaussian Process with Attention](https://proceedings.iclr.cc/paper_files/paper/2024/hash/826aea2253363fe04e8c4991b2a8869e-Abstract-Conference.html)
- [E. Balzani, et al., A probabilistic framework for task-aligned intra- and inter-area neural manifold estimation](https://doi.org/10.48550/arXiv.2209.02816)
- [V. Caro, et al., Modeling Neonatal EEG Using Multi-Output Gaussian Processes](https://ieeexplore.ieee.org/abstract/document/9734069)
- [L. Xu, et al., Prediction for distributional outcomes in high-performance computing input/output variability](https://doi.org/10.1093/jrsssc/qlae001)
- [R.R. Griffiths, Applications of Gaussian Processes at Extreme Lengthscales: From Molecules to Black Holes](https://www.proquest.com/openview/220275b1658d87362aedca7dabca06c5/1?pq-origsite=gscholar&cbl=2026366&diss=y)
- [G. Yang, et al., Self-Evolving Offset-Free Model Predictive Control with Model–Plant Mismatch for Dynamic Working-Point Change Tasks in Industrial Processes](https://pubs.acs.org/doi/10.1021/acs.iecr.2c04642)
- [Ó. García-Hinde, et al., A conditional one-output likelihood formulation for multitask Gaussian processes](https://doi.org/10.1016/j.neucom.2022.08.064)
- [L. Liu, et al., Plane Cascade Aerodynamic Performance Prediction Based on Metric Learning for Multi-Output Gaussian Process Regression](https://doi.org/10.3390/sym15091692)
- [Y. Chai, Uncertainty Quantification for Network Models: A Study in Synthetic Brain Models](https://ore.exeter.ac.uk/articles/thesis/Uncertainty_Quantification_for_Network_Models_A_Study_in_Synthetic_Brain_Models/30454622?file=59088764)
- [A. Khanal, et al., Gaussian Process-Based Extended Kalman Filter for Trajectory Estimation in sUAV Traffic Management](https://doi.org/10.2514/6.2026-0545)
- [D. Özese, et al, Tree-Based Sequential Sampling for Efficient Designs in Package Electrical Analysis](https://ieeexplore.ieee.org/abstract/document/10539229)
- [F. Batsch, Active Learning with Gaussian Processes for the Investigation of Critical Scenarios in Autonomous Driving](https://pureportal.coventry.ac.uk/en/studentTheses/active-learning-with-gaussian-processes-for-the-investigation-of-/)
- [V. Caro, et al., Modeling neonatal EEG using multi-output Gaussian processes](https://repositorio.uchile.cl/handle/2250/186592)
- [Z. Sun, K. Chen, Leveraging Single Tasks for Better Generalization of Multitask Gaussian Process on Multivariate Time Series](https://doi.org/10.21203/rs.3.rs-4839107/v1)
- [F. Tobar, et al., Data Science for Engineers: A Teaching Ecosystem](https://ieeexplore.ieee.org/abstract/document/9418568)
- [F. Tobar, et al., Computationally-efficient initialisation of GPs: The generalised variogram method](https://doi.org/10.48550/arXiv.2210.05394)
- [Q. Xu, et al., Revisiting Nonstationary Kernel Design for Multi-Output Gaussian Processes](https://openreview.net/forum?id=vFfujX5Ygn)

## Books
- [Michael Ludkovski, Jimmy Risk, Gaussian Process Models for Quantitative Finance](https://link.springer.com/book/10.1007/978-3-031-80874-6)


### Used in code
- https://github.com/jdjmoon/TRF
- https://github.com/ErickRosete/Multivariate_regression
- https://github.com/clara-risk/fire_weather_interpolate
- https://github.com/becre2021/multichannels-corrnp
- https://github.com/ArthurLeroy/MAGMAclust
- https://github.com/nicdel-git/master_thesis

