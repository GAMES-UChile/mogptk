import numpy as np
import pandas as pd          # TODO: remove dependency on sklearn?
from sklearn import metrics  # TODO: remove dependency on sklearn?

def mean_absolute_percentage_error(y_true, y_pred):
    idx = np.nonzero(y_true)
    y_true = y_true[idx]
    y_pred = y_pred[idx]
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# TODO: use relative error per channel
def errors(*models, **kwargs):
    """
    Return error metrics for given models.

    errors will returns error measures (MAE, MSE, MAPE) for the model by comparing the deleted observations from the predicted means.
    The predicted values are interpolated linearly to match the X position of the delete dobservations.
    However if a latent function is defined in the data this will be used as the true values, which gets rid of the imposed Gaussian error on the observations.

    Args:
        model_list: Iterable with mogptk models to evaluate.
    
    Returns:
        errors (dic): Dictionary with lists of ndarrays containing different error metrics per model, per channel.

        The dictionary has three keys, 'model' which contains model name; 'MAE' contains mean absolute error; 'MSE' mean squared error; 'MAPE' mean absolute percentage error.
        


    """
    all_obs = False
    if "all_obs" in kwargs:
        all_obs = kwargs["all_obs"]
    output = False
    if "print" in kwargs:
        output = kwargs["print"]

    errors = []
    for model in models:
        if model.get_input_dims() != 1:
            raise Exception("cannot (yet) estimate errors when using multi input dimensions")

        Y_true = np.empty(0)
        Y_pred = np.empty(0)
        for channel in model.data:
            if len(channel.X_pred) == 0:
                continue

            if all_obs:
                x, y_true = channel.get_data()
            else:
                x, y_true = channel.get_test_data()

            if len(x) > 0:
                if channel.F != None:
                    y_true = channel.F(x) # use exact latent function to remove imposed Gaussian error on data points

                y_pred = np.interp(x, channel.X_pred.reshape(-1), channel.Y_mu_pred) # TODO: multi input dims

                Y_true = np.append(Y_true, y_true)
                Y_pred = np.append(Y_pred, y_pred)
            
        errors.append({
            "model": model.name,
            "MAE": metrics.mean_absolute_error(Y_true, Y_pred),
            "MSE": metrics.mean_squared_error(Y_true, Y_pred),
            "MAPE": mean_absolute_percentage_error(Y_true, Y_pred),
        })

    if output:
        df = pd.DataFrame(errors)
        df.set_index('model', inplace=True)
        display(df)
    else:
        return errors


def test_errors(*models, x_test, y_test, raw_errors=False):
    """
    Return test errors given a model and test set.

    The function assumes all models have been trained and all models
    share equal number of inputs and outputs (channels).

    Args:
        models (mogptk.model): Trained model to evaluate, can be more than one

        x_test (list): List of numpy arrays with the inputs of the test set.
            Length is the output dimension.

        y_test (list): List of numpy array with the true ouputs of test set.
            Length is the output dimension.

        raw_errors (bool): If true returns for each model a list is returned
            with the errors of each channel (y_true - y_pred).
            If false returns for each model a list of 4 arrays with the
            mean absolute error (MAE), range-normalized mean absolute error (nMAE),
            root mean squared error (RMSE) and range-normalized root mean 
            squared error (nRMSE) for each channel.

    Returns:
        List with length equal to the number of models, each element
        contains a list of length of the output dim and each
        element is an array with the errors.

    Example:
        Given model1, model2, x_test, y_test of correct format.

        >>> errors = mogptk.test_errors(model1, model2, x_test, y_test)
        >>> errors[i][j]
        numpy array with errors from model 'i' at channel 'j'
    """

    error_per_model = []

    for model in models:

        n_channels = model.dataset.get_output_dims()

        if n_channels==1:
            if not isinstance(y_test, list):
                y_test = [y_test]

        error_per_channel = []

        # print([a.std() for a in y_test])

        # predict with model
        y_pred, _, _ = model.predict(x_test)

        for i in range(n_channels):
            errors = y_test[i] - y_pred[i]
            # if only error values
            if raw_errors:
                error_per_channel.append(errors)

            # composite errors
            else:
                y_range = y_test[i].max() - y_test[i].min()

                mae = np.abs(errors).mean()
                nmae = mae / y_range
                rmse = np.sqrt((errors**2).mean())
                nrmse = rmse / y_range
                
                error_per_channel.append(np.array([mae, nmae, rmse, nrmse]))

        error_per_model.append(error_per_channel)

    return error_per_model

