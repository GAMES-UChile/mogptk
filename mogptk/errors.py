import numpy as np
import pandas as pd

def _check_arrays(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    idx = np.nonzero(y_true)
    return y_true[idx], y_pred[idx]

def mean_absolute_error(y_true, y_pred):
    """
    Mean Absolute Error (MAE) between the true and the predicted values.
    """
    y_true, y_pred = _check_arrays(y_true, y_pred)
    return np.mean(np.abs(y_true - y_pred))

def root_mean_squared_error(y_true, y_pred):
    """
    Root Mean Squared Error (RMSE) between the true and the predicted values.
    """
    y_true, y_pred = _check_arrays(y_true, y_pred)
    return np.sqrt(np.mean((y_true - y_pred)^2))

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Mean Absolute Percentage Error (MAPE) between the true and the predicted values.
    """
    y_true, y_pred = _check_arrays(y_true, y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# TODO: use relative error per channel
def prediction_error(dataset, all_obs=False, disp=False):
    """
    Return error metrics for the predictions on the dataset (MAE, RMSE, MAPE) by comparing the removed observations from the predicted means. The predicted values are interpolated linearly to match the X position of the removed observations. However, if a latent function is defined for the data then this will be used instead as the true values.

    Args:
        dataset (mogptk.DataSet): Data set containing predictions to evaluate.
        all_obs (boolean, optional): Use all observations for error calculation instead of using only removed observations.
        disp (boolean, optional): Display errors instead of returning them.
    
    Returns:
        errors (list): List of dictionaries per model, containing numpy.ndarrays for each error metric. The dictionary has the following keys: Name which contains the model name, MAE contains the mean absolute error, MAPE the mean absolute percentage error, and RMSE the root mean squared error.

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
            "MAE": mean_absolute_error(Y_true, Y_pred),
            "MAPE": mean_absolute_percentage_error(Y_true, Y_pred),
            "RMSE": root_mean_squared_error(Y_true, Y_pred),
        })

    if disp:
        df = pd.DataFrame(errors)
        df.set_index('Name', inplace=True)
        display(df)
    else:
        return errors

def error(*models, X=None, Y=None, per_channel=False, disp=False):
    """
    Return test errors given a model and a test set. The function assumes all models have been trained and all models share equal numbers of inputs and outputs (channels). If X and Y are not passed and only one model is given, than the dataset's test data of the model is used.

    Args:
        models (list of mogptk.model): Trained model to evaluate, can be more than one
        X (list): List of numpy arrays with the inputs of the test set. Length is the output dimension.
        Y (list): List of numpy arrays with the true ouputs of test set. Length is the output dimension.
        per_channel (boolean, optional): Return averages over channels instead of over models.
        disp (boolean, optional): Display errors instead of returning them.

    Returns:
        List with length equal to the number of models (times the number of channels if per_channel is given), each element containing a dictionary with the following keys: Name which contains the model name, MAE contains the mean absolute error, MAPE the mean absolute percentage error, and RMSE the root mean squared error.

    Example:
        >>> errors = mogptk.test_errors(model1, model2, X_true, Y_true, per_channel=True)
        >>> errors[i][j]  # error metrics for model i and channel j
    """
    if len(models) == 0:
        raise ValueError("must pass models")
    elif len(models) == 1 and X is None and Y is None:
        X, Y = models[0].dataset.get_test_data()
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

    Y_true = Y
    errors = []
    for model in models:
        _, Y_pred, _, _ = model.predict(X)
        if per_channel:
            model_errors = []
            for i in range(model.dataset.get_output_dims()):
                model_errors.append({
                    "Name": model.name + " channel " + str(i),
                    "MAE": mean_absolute_error(Y_true[i], Y_pred[i]),
                    "MAPE": mean_absolute_percentage_error(Y_true[i], Y_pred[i]),
                    "RMSE": root_mean_squared_error(Y_true[i], Y_pred[i]),
                })
            errors.append(model_errors)
        else:
            Ys_true = [item for sublist in Y_true for item in sublist]
            Ys_pred = [item for sublist in Y_pred for item in sublist]
            errors.append({
                "Name": model.name,
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

