import numpy as np
import pandas as pd
from sklearn import metrics  # TODO: remove dependency on sklearn?

def mean_absolute_percentage_error(y_true, y_pred):
    idx = np.nonzero(y_true)
    y_true = y_true[idx]
    y_pred = y_pred[idx]
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# TODO: use relative error per channel
def errors(dataset, all_obs=False, disp=False):
    """
    Return error metrics for given models (MAE, MSE, MAPE) by comparing the deleted observations from the predicted means. The predicted values are interpolated linearly to match the X position of the delete dobservations. However if a latent function is defined in the data this will be used as the true values, which gets rid of the imposed Gaussian error on the observations.

    Args:
        dataset (mogptk.DataSet): DataSet containing predictions to evaluate.
        all_obs (boolean, optional): Use all observations for error calculation instead of using only removed observations.
        disp (boolean, optional): Display errors instead of returning them.
    
    Returns:
        errors (dict): Dictionary with lists of ndarrays containing different error metrics per model, per channel. The dictionary has three keys, 'model' which contains model name; 'MAE' contains mean absolute error; 'MSE' mean squared error; 'MAPE' mean absolute percentage error.

    """
    if dataset.get_output_dims() == 0:
        raise ValueError("dataset is empty")

    names = dataset[0].get_prediction_names()
    if len(names) == 0:
        raise ValueError("there are no predictions")

    errors = []
    for name in names:
        Y_true = np.empty(0)
        Y_pred = np.empty(0)
        for channel in dataset:
            if channel.get_input_dims() != 1:
                raise ValueError("can only estimate errors on one dimensional input data")
            if name not in channel.get_prediction_names():
                continue

            if all_obs:
                xt, _ = channel.get_data(transformed=True)
                x, y_true = channel.get_data()
            else:
                xt, _ = channel.get_test_data(transformed=True)
                x, y_true = channel.get_test_data()
            if channel.F is not None:
                y_true = channel.F(x) # use exact latent function

            xt_pred, _, _, _ = channel.get_prediction(name, transformed=True)
            _, mu, _, _ = channel.get_prediction(name)
            y_pred = np.interp(xt[:,0], xt_pred[:,0], mu)

            Y_true = np.append(Y_true, y_true)
            Y_pred = np.append(Y_pred, y_pred)
            
        errors.append({
            "Name": name,
            "MAE": metrics.mean_absolute_error(Y_true, Y_pred),
            "MSE": metrics.mean_squared_error(Y_true, Y_pred),
            "MAPE": mean_absolute_percentage_error(Y_true, Y_pred),
        })

    if disp:
        df = pd.DataFrame(errors)
        df.set_index('Name', inplace=True)
        display(df)
    else:
        return errors

def errors_at(*models, X, Y, simple=False, per_channel=False, disp=False):
    """
    Return test errors given a model and test set. The function assumes all models have been trained and all models share equal number of inputs and outputs (channels).

    Args:
        models (list of mogptk.model): Trained model to evaluate, can be more than one
        X (list): List of numpy arrays with the inputs of the test set. Length is the output dimension.
        Y (list): List of numpy arrays with the true ouputs of test set. Length is the output dimension.
        simple (boolean, optional): If True returns for each model the error as (Y_pred - Y_true). If False returns for each model the mean absolute error (MAE), normalized mean absolute error (nMAE), root mean squared error (RMSE), normalized root mean squared error (nRMSE), and the mean absolute percentage error (MAPE) for each channel.
        per_channel (boolean, optional): Return averages over channels instead of over models.
        disp (boolean, optional): Display errors instead of returning them.

    Returns:
        List with length equal to the number of models, each element contains a list of length of the output dim and each element is an array with the errors.

    Example:
        Given model1, model2, x_test, y_test of correct format.

        >>> errors = mogptk.test_errors(model1, model2, X_true, Y_true)
        >>> errors[i][j]  # numpy array with errors from model 'i' at channel 'j'
    """
    if simple and disp:
        raise ValueError("cannot set both simple and disp")
    if len(models) == 0:
        raise ValueError("must pass models")
    
    output_dims = models[0].dataset.get_output_dims()
    if not isinstance(X, list):
        X = [X] * output_dims
    if not isinstance(Y, list):
        Y = [Y] * output_dims
    for model in models[1:]:
        if model.dataset.get_output_dims() != output_dims:
            raise ValueError("all models must have the same number of channels")
    if len(X) != output_dims or len(X) != len(Y):
        raise ValueError("X and Y must be lists with as many entries as channels")

    X_true = X
    Y_true = Y
    errors = []
    for model in models:
        Y_pred, _, _ = model.predict(X_true)
        model_errors = []
        for i in range(model.dataset.get_output_dims()):
            err = Y_pred[i] - Y_true[i]
            if simple:
                model_errors.append(err)
            else:
                Y_range = Y_true[i].max() - Y_true[i].min()
                model_errors.append({
                    "Name": model.name + " channel " + str(i),
                    "MAE": metrics.mean_absolute_error(Y_true[i], Y_pred[i]),
                    "nMAE": metrics.mean_absolute_error(Y_true[i], Y_pred[i]) / Y_range,
                    "RMSE": metrics.mean_squared_error(Y_true[i], Y_pred[i], squared=False),
                    "nRMSE": metrics.mean_squared_error(Y_true[i], Y_pred[i], squared=False) / Y_range,
                    "MAPE": mean_absolute_percentage_error(Y_true[i], Y_pred[i]),
                })
        errors.append(model_errors)

    if disp:
        df = pd.DataFrame([item for sublist in errors for item in sublist])
        df.set_index('Name', inplace=True)
        display(df)
    else:
        return errors

