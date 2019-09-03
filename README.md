[Documentation](https://games-uchile.github.io/MultiOutputGP-Toolbox/)

# MultiOutputGP-Toolbox
This repository provides a toolbox to perform multi-output GP regression with kernels that are designed to utilize correlation information among channels in order to better model signals. The toolbox is mainly targeted to time-series, and includes plotting functions for the case of single input with multiple outputs (time series with several channels). The main kernel corresponds to Multi Output Spectral Mixture Kernel, which correlates every pair of data points (irrespective of their channel of origin) to model the signals. This kernel is specified in detail in the following publication: G. Parra, F. Tobar, Spectral Mixture Kernels for Multi-Output Gaussian Processes, Advances in Neural Information Processing Systems, 2017

Proceedings link: http://papers.nips.cc/paper/7245-spectral-mixture-kernels-for-multi-output-gaussian-processes

The kernel learns the cross-channel correlations of the data, so it is particularly well-suited for the task of signal reconstruction in the event of sporadic data loss.

All other included kernels can be derived from the Multi Output Spectral Mixture kernel by restricting some parameters or applying some transformations.

One of the main advantages of the present toolbox is the GPU support, which enables the user to train models through TensorFlow, speeding computations significantly. It also includes sparse-variational GP regression functionality, to decrease computation time even further.

## Installation guide
Make sure you have Python 3.6 and `pip` installed, and run the following command:

```
pip install git+https://github.com/GAMES-UChile/MultiOutputGP-Toolbox
```

## Notes on requirements
It seems that the latest tensorflow packages can be found in the anaconda channel, so using the
following command to create a new conda environment yields the easiest results:
	conda create -n TFGPU -c anaconda scipy scikit-learn matplotlib seaborn tensorflow-gpu==1.12 

This command creates a new env named TFGPU with tensorflow-gpu 1.12 (as of this writing).

## License
Released under the [MIT license](LICENSE).
