import logging
import dill
from .plot import plot_spectrum
from .data import Data
from .model import model
from .kernels import SpectralMixture, sm_init, MultiOutputSpectralMixture, SpectralMixtureLMC, ConvolutionalGaussian, CrossSpectralMixture, Noise
from .kernels.conv_old import ConvolutionalGaussianOLD
import numpy as np
import gpflow
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import norm

# TODO
def _estimate_noise_var(data):
    """
    Estimate noise variance with a tenth of the channel variance
    """

    output_dims = len(data)

    noise_var = np.zeros(output_dims)

    for i, channel in enumerate(data):
        noise_var[i] = (channel.Y).var() / 30

    return noise_var

