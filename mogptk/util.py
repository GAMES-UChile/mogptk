import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

def mean_absolute_error(y_true, y_pred):
    """
    Mean Absolute Error (MAE) between the true and the predicted values.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Mean Absolute Percentage Error (MAPE) between the true and the predicted values.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    idx = 1e-6 < y_true
    y_true, y_pred = y_true[idx], y_pred[idx]
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error (sMAPE) between the true and the predicted values.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    idx = 1e-6 < y_true
    y_true, y_pred = y_true[idx], y_pred[idx]
    return np.mean(np.abs((y_true - y_pred) / (y_true + y_pred))) * 200.0

def mean_squared_error(y_true, y_pred):
    """
    Mean Squared Error (MSE) between the true and the predicted values.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean((y_true - y_pred)**2)

def root_mean_squared_error(y_true, y_pred):
    """
    Root Mean Squared Error (RMSE) between the true and the predicted values.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sqrt(np.mean((y_true - y_pred)**2))

# TODO: use relative error per channel
def error(*models, X=None, Y=None, per_channel=False, transformed=False, disp=False):
    """
    Return test errors given a model and a test set. The function assumes all models have been trained and all models share equal numbers of inputs and outputs (channels). If X and Y are not passed than the dataset's test data are used.

    Args:
        models (list of mogptk.model.Model): Trained model to evaluate, can be more than one
        X (list): List of numpy arrays with the inputs of the test set. Length is the output dimension.
        Y (list): List of numpy arrays with the true ouputs of test set. Length is the output dimension.
        per_channel (boolean): Return averages over channels instead of over models.
        transformed (boolean): Use the transformed data to calculate the error.
        disp (boolean): Display errors instead of returning them.

    Returns:
        List with length equal to the number of models (times the number of channels if per_channel is given), each element containing a dictionary with the following keys: Name which contains the model name, MAE contains the mean absolute error, MAPE the mean absolute percentage error, and RMSE the root mean squared error.

    Example:
        >>> errors = mogptk.error(model1, model2, X_true, Y_true, per_channel=True)
        >>> errors[i][j]  # error metrics for model i and channel j
    """
    if len(models) == 0:
        raise ValueError("must pass models")
    elif X is None and Y is None:
        X, Y = models[0].dataset.get_test_data(transformed=transformed)
        for model in models[1:]:
            X2, Y2 = model.dataset.get_test_data(transformed=transformed)
            if len(X) != len(X2) or not all(np.array_equal(X[j],X2[j]) for j in range(len(X))) or not all(np.array_equal(Y[j],Y2[j]) for j in range(len(X))):
                raise ValueError("all models must have the same data set for testing, otherwise explicitly provide X and Y")
        if sum(x.size for x in X) == 0:
            raise ValueError("models have no test data")
    elif X is None and Y is not None or X is not None and Y is None:
        raise ValueError("X and Y must both be set or omitted")

    output_dims = models[0].dataset.get_output_dims()
    for model in models[1:]:
        if model.dataset.get_output_dims() != output_dims:
            raise ValueError("all models must have the same number of channels")
    if not isinstance(X, list):
        X = [X] * output_dims
    if not isinstance(Y, list):
        Y = [Y] * output_dims
    if len(X) != output_dims or len(X) != len(Y):
        raise ValueError("X and Y must be lists with as many entries as channels")

    Y_true = Y
    errors = []
    for k, model in enumerate(models):
        name = model.name
        if name is None:
            name = "Model " + str(k+1)

        X, Y_pred, _, _ = model.predict(X, transformed=transformed)
        if len(model.dataset) == 1:
            Y_pred = [Y_pred]

        if per_channel:
            model_errors = []
            for j in range(model.dataset.get_output_dims()):
                model_errors.append({
                    "Name": name + " channel " + str(j+1),
                    "MAE": mean_absolute_error(Y_true[j], Y_pred[j]),
                    "MAPE": mean_absolute_percentage_error(Y_true[j], Y_pred[j]),
                    "RMSE": root_mean_squared_error(Y_true[j], Y_pred[j]),
                })
            errors.append(model_errors)
        else:
            Ys_true = np.concatenate(Y_true, axis=0)
            Ys_pred = np.concatenate(Y_pred, axis=0)
            errors.append({
                "Name": name,
                "MAE": mean_absolute_error(Ys_true, Ys_pred),
                "MAPE": mean_absolute_percentage_error(Ys_true, Ys_pred),
                "RMSE": root_mean_squared_error(Ys_true, Ys_pred),
            })

    if disp:
        if per_channel:
            df = pd.DataFrame([item for sublist in errors for item in sublist])
        else:
            df = pd.DataFrame(errors)
        df.set_index('Name', inplace=True)
        display(df)
    else:
        return errors

def plot_spectrum(means, scales, dataset=None, weights=None, noises=None, method='LS', maxfreq=None, log=False, n=10000, titles=None, show=True, filename=None, title=None):
    """
    Plot spectral Gaussians of given means, scales and weights.
    """
    if means.ndim == 2:
        means = np.expand_dims(means, axis=2)
    if scales.ndim == 2:
        scales = np.expand_dims(scales, axis=2)
    if isinstance(weights, np.ndarray) and weights.ndim == 1:
        weights = np.expand_dims(weights, axis=1)
    if isinstance(maxfreq, np.ndarray) and maxfreq.ndim == 1:
        maxfreq = np.expand_dims(maxfreq, axis=1)

    if means.ndim != 3:
        raise ValueError('means and scales must have shape (mixtures,output_dims,input_dims)')
    if means.shape != scales.shape:
        raise ValueError('means and scales must have the same shape (mixtures,output_dims,input_dims)')
    if noises is not None and (noises.ndim != 1 or noises.shape[0] != means.shape[1]):
        raise ValueError('noises must have shape (output_dims,)')
    if dataset is not None and len(dataset) != means.shape[1]:
        raise ValueError('means and scales must have %d output dimensions' % len(dataset))

    mixtures = means.shape[0]
    output_dims = means.shape[1]
    input_dims = means.shape[2]

    if isinstance(weights, np.ndarray) and (weights.ndim != 2 or weights.shape[0] != mixtures or weights.shape[1] != output_dims):
        raise ValueError('weights must have shape (mixtures,output_dims)')
    elif not isinstance(weights, np.ndarray):
        weights = np.ones((mixtures,output_dims))
    if isinstance(maxfreq, np.ndarray) and (maxfreq.ndim != 2 or maxfreq.shape[0] != output_dims or maxfreq.shape[1] != input_dims):
        raise ValueError('maxfreq must have shape (output_dims,input_dims)')


    h = 4.0*output_dims
    fig, axes = plt.subplots(output_dims, input_dims, figsize=(12,h), squeeze=False, constrained_layout=True)
    if title is not None:
        fig.suptitle(title, y=(h+0.8)/h, fontsize=18)
    
    for j in range(output_dims):
        for i in range(input_dims):
            x_low = max(0.0, norm.ppf(0.01, loc=means[:,j,i], scale=scales[:,j,i]).min())
            x_high = norm.ppf(0.99, loc=means[:,j,i], scale=scales[:,j,i]).max()

            if dataset is not None:
                maxf = maxfreq[j,i] if maxfreq is not None else None
                dataset[j].plot_spectrum(ax=axes[j,i], method=method, transformed=True, n=n, log=False, maxfreq=maxf)
                x_low = axes[j,i].get_xlim()[0]
                x_high =  axes[j,i].get_xlim()[1]
            if maxfreq is not None:
                x_high = maxfreq[j,i]

            psds = []
            x = np.linspace(x_low, x_high, n)
            psd_total = np.zeros(x.shape)
            for q in range(mixtures):
                psd = weights[q,j] * norm.pdf(x, loc=means[q,j,i], scale=scales[q,j,i])
                axes[j,i].axvline(means[q,j,i], ymin=0.001, ymax=0.05, lw=3, color='r')
                psd_total += psd
                psds.append(psd)
            if noises is not None:
                psd_total += noises[j]**2

            for psd in psds:
                psd /= psd_total.sum()*(x[1]-x[0]) # normalize
                axes[j,i].plot(x, psd, ls='--', c='b')
            psd_total /= psd_total.sum()*(x[1]-x[0]) # normalize
            axes[j,i].plot(x, psd_total, ls='-', c='b')

            y_low = 0.0
            if log:
                x_low = max(x_low, 1e-8)
                y_low = 1e-8
            _, y_high = axes[j,i].get_ylim()
            y_high = max(y_high, 1.05*psd_total.max())
           
            axes[j,i].set_xlim(x_low, x_high)
            axes[j,i].set_ylim(y_low, y_high)
            axes[j,i].set_yticks([])
            if titles is not None:
                axes[j,i].set_title(titles[j])
            if log:
                axes[j,i].set_xscale('log')
                axes[j,i].set_yscale('log')

    axes[output_dims-1,i].set_xlabel('Frequency')

    legends = []
    if dataset is not None:
        legends.append(plt.Line2D([0], [0], ls='-', color='k', label='Data (LombScargle)'))
    legends.append(plt.Line2D([0], [0], ls='-', color='b', label='Model'))
    legends.append(plt.Line2D([0], [0], ls='-', color='r', label='Peak location'))
    fig.legend(handles=legends)

    if filename is not None:
        plt.savefig(filename+'.pdf', dpi=300)
    if show:
        plt.show()
    return fig, axes
