from model import mogp_model
import numpy as np
import copy
import time
import tensorflow as tf

#First, a simple example for 3 sine curves.
def latent(x):
    return np.sin(6*x)

#To make results reproducible.
np.random.seed(20)

X_list = []
#We construct a new set of observations, noting that observations must be sorted according to X axis.
for new_observation in range(3):
    X_list.append(np.sort(np.random.rand(100)))
y_list = []
for new_observation in range(3):
    y_list.append(np.sin(6*X_list[new_observation]) + np.random.standard_t(3, X_list[new_observation].shape)*0.3)

#We define the model, stating the number of components Q.
model = mogp_model(3)
#Since we know the functions that we're trying to approximate we can declare a
#set of latent functions. The first argument is the set of functions, and the
#second argument contains the corresponding channels.
model.define_latent_functions([latent, latent, latent], [0,1,2])

#This method removes slabs from the time series of each channel, in order to mimic
#temporary sensor failure.
X_new, y_new, X_deleted, y_deleted = model.remove_slabs(X_list, y_list)

#The model needs a specific format, so we transform our observations
#to comply with it.
# X, y = model.transform_lists_into_multioutput_format(X_new, y_new)
#With this function we feed the model our training data.
model.add_training_data(X_new, y_new)

#We could add the removed points to plot them in blue.
model.add_extra_observations(X_deleted,y_deleted,[0,1,2])

##################### One option #######################
# #We can train the model with random parameters
# model.build_model()
# #And then optimise it
# model.optimize(iters=1000, display=True, anchor=True)
#######################################################


################### Pick one ##########################
# #Or we could use one of the pre-made optimisation heuristics
model.optimization_heuristic_zero()
# model.optimization_heuristic_one()
# model.optimization_heuristic_two()
# model.optimization_heuristic_three()
#######################################################


#################### Or load ##########################
# #We could also load an already trained model.
# model.load("simple_sines")
#######################################################

print("Negative log likelihood: ", -model.compute_log_likelihood())
#We can automatically generate a prediction interval.
#First argument is the point resolution (number of points).
#The second argument contains the desired channels to predict upon.
X_pred = model.predict_interval(300, [0,1,2])

#We can compute the MAE over some data, in this case we'll compute
#over the complete dataset (including removed points which the
#gp predicts). Note that this performs a prediction based on X_list.
mae = model.compute_mae(X_list, y_list)
print("MAE over original data = ", mae)

#We perform another prediction over X_pred.
Y_pred, STD_pred = model.predict(X_pred)

#We can save plots for our previous prediction. (Predictions made with predict)
model.make_plots("experiment_load.png")

#We can save the trained model.
model.save("simple_sines")
#We can inspect the trained model parameters.
parameters = model.as_pandas_table()
print(parameters)
