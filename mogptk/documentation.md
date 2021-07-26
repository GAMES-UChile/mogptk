MOGPTK is a Python toolkit for multi output Gaussian processes. It contains data handling classes and different multi output models to facilitate the training and prediction of multi channel data sets. It provides a complete toolkit to improve the training of MOGPs, where it allows for easy detrending or transformation of the data (such as taking the natural logarithm), functions that aid in adding gaps to the data or specifying which range should be predicted, and more.

The following four MOGP models have been implemented:

- Multi Output Spectral Mixture (MOSM) as proposed by [1]
- Cross Spectral Mixture (CSM) as proposed by [2]
- Convolutional Gaussian (CONV) as proposed by [3]
- Spectral Mixture - Linear Model of Coregionalization (SM-LMC) as proposed by combining SM [4] and LMC [5]

Each model has specialized methods for parameter estimation to improve training. For example, the MOSM model can estimate its parameters using the training of spectral mixture kernels for each of the channels individually. It can also estimate its parameters using Bayesian Nonparametric Spectral Estimation (BNSE) [6]. After parameter estimation, the training of the model will converge more rapidly and there is a higher chance it will converge at all. Additionally, some models allow the interpretation of the parameters by investigating the cross-correlation parameters between the channels to better interpret what the learned parameters actually mean.

- [1] G. Parra and F. Tobar, "Spectral Mixture Kernels for Multi-Output Gaussian Processes", Advances in Neural Information Processing Systems 31, 2017
- [2] K.R. Ulrich et al, "GP Kernels for Cross-Spectrum Analysis", Advances in Neural Information Processing Systems 28, 2015
- [3] M.A. Álvarez and N.D. Lawrence, "Sparse Convolved Multiple Output Gaussian Processes", Advances in Neural Information Processing Systems 21, 2009
- [4] A.G. Wilson and R.P. Adams, "Gaussian Process Kernels for Pattern Discovery and Extrapolation", International Conference on Machine Learning 30, 2013
- [5] P. Goovaerts, "Geostatistics for Natural Resource Evaluation", Oxford University Press, 1997
- [6] F. Tobar, "Bayesian Nonparametric Spectral Estimation", Advances in Neural Information Processing Systems 32, 2018

## Installation
Make sure you have at least Python 3.6 and `pip` installed, and run the following command:

```
pip install mogptk
```

## Glossary
- **GP**: Gaussian process, see [Gaussian Processes for Machine Learning](http://www.gaussianprocess.org/gpml/) by C.E. Rasmussen and C.K.I. Williams.
- **M**: the number of channels (i.e. output dimensions)
- **N**: the number of data points
- **Q**: the number of components to use for a kernel. Each component is added additively to the main kernel
- **MOSM**: Multi-output spectral mixture kernel, see [Spectral Mixture Kernels for Multi-Output Gaussian Processes](https://arxiv.org/abs/1709.01298) by G. Parra and F. Tobar.
- **CSM**: Cross spectral mixture kernel, see [GP Kernels for Cross-Spectrum Analysis](https://papers.nips.cc/paper/5966-gp-kernels-for-cross-spectrum-analysis) by K.R. Ulrich et al.
- **CONV**: Convolution Gaussian kernel, see [Sparse Convolved Multiple Output Gaussian Processes](https://arxiv.org/abs/0911.5107) by M.A. Álvarez and N.D. Lawrence.
- **SM-LMC**: Spectral mixture linear model of coregionalization kernel, see [Gaussian Process Kernels for Pattern Discovery and Extrapolation](https://arxiv.org/abs/1302.4245) by A.G. Wilson and R.P. Adams and the book "Geostatistics for Natural Resource Evaluation" by P. Goovaerts.

## Data handling
### Transformations
The transformation classes allow transforming the dependent data (Y axis) to be transformed. Each class implements the following functions:

- **set_data(data)**: pass the Data class in case the transformer uses that data to calculate the transformation
- **forward(y, x=None)** returns y: does a forward transformation where x has shape (n,input\_dims) and y has shape (n,)
- **backward(y, x=None)** returns y: does a backward transformation (invert) where x has shape (n,input\_dims) and y has shape (n,)

## Using the GPU
If a GPU is available through CUDA it will be automatically used in tensor calculations, and may speed up training significantly. To get more information about whether CUDA is supported or which GPU is used, as well as more control over which CPU or GPU to use, see [mogptk.gpr.config](https://games-uchile.github.io/mogptk/gpr/config.html).

## Advice on training 

## Visualization and interpretation
