import numpy as np
from sklearn import metrics
from sklearn.utils import check_array

def mean_absolute_percentage_error(y_true, y_pred):
    idx = np.nonzero(y_true)
    y_true = y_true[idx]
    y_pred = y_pred[idx]
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def errors(model_list, **kwargs):
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

    errors = {
        "model": [],
        "MAE": [],
        "MSE": [],
        "MAPE": [],
    }
    for model in model_list:
        Y_true = np.empty(0)
        Y_pred = np.empty(0)
        for channel in range(model.data.get_output_dims()):
            if all_obs:
                x, y_true = model.data.get_all_obs(channel)
            else:
                x, y_true = model.data.get_del_obs(channel)

            if len(x) > 0:
                if channel in model.data.F:
                    y_true = model.data.F[channel](x) # use exact latent function to remove imposed Gaussian error on data points

                y_pred = np.interp(x, model.X_pred[channel], model.Y_mu_pred[channel])

                Y_true = np.append(Y_true, y_true)
                Y_pred = np.append(Y_pred, y_pred)
            
        errors["model"].append(model.name)
        errors["MAE"].append(metrics.mean_absolute_error(Y_true, Y_pred))
        errors["MSE"].append(metrics.mean_squared_error(Y_true, Y_pred))
        errors["MAPE"].append(mean_absolute_percentage_error(Y_true, Y_pred))

    if output:
        import pandas as pd
        df = pd.DataFrame(errors)
        df.set_index('model', inplace=True)
        display(df)
    else:
        return errors

