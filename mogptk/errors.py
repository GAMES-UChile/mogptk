import numpy as np
import pandas as pd

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
        if sum(x.shape[0] for x in X) == 0:
            raise ValueError("models have no test data")

    elif X is None and Y is not None or X is not None and Y is None:
        raise ValueError("X and Y must both be set or omitted")

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
    for k, model in enumerate(models):
        name = model.name
        if name is None:
            name = "Model " + str(k+1)

        Y_pred, _, _ = model.predict(X, transformed=transformed)
        if per_channel:
            model_errors = []
            for i in range(model.dataset.get_output_dims()):
                model_errors.append({
                    "Name": name + " channel " + str(i+1),
                    "MAE": mean_absolute_error(Y_true[i], Y_pred[i]),
                    "MAPE": mean_absolute_percentage_error(Y_true[i], Y_pred[i]),
                    "RMSE": root_mean_squared_error(Y_true[i], Y_pred[i]),
                })
            errors.append(model_errors)
        else:
            Ys_true = np.array([item for sublist in Y_true for item in sublist])
            Ys_pred = np.array([item for sublist in Y_pred for item in sublist])
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

