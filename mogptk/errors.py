import numpy as np
from sklearn import metrics
from sklearn.utils import check_arrays

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = check_arrays(y_true, y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def errors(*args, **kwargs):
    """errors will returns error measures (MAE, MSE, ...) for the model by comparing the deleted observations from the predicted means. The predicted values are interpolated linearly to match the X position of the delete dobservations."""
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
    }
    for model in args:
        Y_true = np.empty(0)
        Y_pred = np.empty(0)
        for channel in range(model.data.get_output_dims()):
            if all_obs:
                x, y_true = model.data.get_all_obs(channel)
            else:
                x, y_true = model.data.get_del_obs(channel)

            if len(x) > 0:
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

