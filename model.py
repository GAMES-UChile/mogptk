import gpflow
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns
import tensorflow as tf
from mosm.multi_spectralmixture_blocked import MultiSpectralMixtureBlocked as MOSM
from mosm.single_spectralmixture import SpectralMixture, sm_unravel_hp
from collections import Counter
import scipy.stats
import random
import scipy.signal as signal
from bnse import *
from skopt.space import Real, Integer
from skopt import gp_minimize
import copy
import time
from sklearn.metrics import mean_absolute_error
from itertools import accumulate

class mosm_model:
    #At model construction we define the number of components and the optimizer
    def __init__(self, number_of_components_q, optimizer = 'L-BFGS-B'):
        self.Q = number_of_components_q
        self.X_extra = {}
        self.y_extra = {}
        self.latent_functions = {}
        self.iterations = 1000
        self.optimization_method = optimizer

############################# Add training data to the model ###############################
    #Packages the data according to the expected multioutput representation.
    #This method takes 2 lists of numpy arrays or lists, named X and y.
    #The pairs {X[i][j], y[i][j]} from j = 0 to j = X[i].shape[0] represent the i-th channel observations.
    #Note that in the multi-dimensional input case each element for a given channel in X
    #is a list or numpy array itself.
    def transform_lists_into_multioutput_format(self, X, y):
        X_multi_output = []
        y_multi_output = []
        for channel in range(len(X)):
            for element in range(len(X[channel])):
                current_x = [channel]
                current_y = [y[channel][element]]
                if(isinstance(X[channel][element], (list, np.ndarray))):
                    for x_i in X[channel][element]:
                        current_x.append(x_i)
                else:
                    current_x.append(X[channel][element])
                X_multi_output.append(current_x)
                y_multi_output.append(current_y)
        return np.array(X_multi_output), np.array(y_multi_output).reshape(-1,1)
    #Takes an X,y pair in the expected multi-output format, which is as
    #follows:
    #For X:
    #channel_id, features
    #For y:
    #Observation
    #To clarify, if we have 2 channels, 3 samples (x0,y0), (x1,y1) and (x2,y2) for
    #the first channel and 2 samples (x3,y3) and (x4,y4) for the second channel,
    #the expected representation would be:
    #For X:
    #np.array([[0,x0]
    #[0,x1]
    #[0,x2]
    #[1,x3]
    #[1,x4]]) -> A vector of stacked numpy array (channel, features) elements.
    #In this example the Xs only have 1 feature, namely the x_i.
    #For y:
    #np.array([[y0]
    #[y1]
    #[y2]
    #[y3]
    #[y4]]) -> A vector of stacked numpy array observations.
    #Which is rather cumbersome. In order to make this more palatable, the function
    #transform_lists_into_multioutput_format is included. This function constructs the
    #expected format from a simpler one. For the previous example the simpler format
    #would be the following:
    #For X:
    #[np.array([x0,x1,x2]),np.array([x3,x4])] -> A list of numpy arrays.
    #For y:
    #[np.array([y0,y1,y2]), np.array([y3,y4])] -> A list of numpy arrays.
    #Note that observations must be ordered increasingly by the x-axis.
    def add_training_data(self, X, y):
        self.X_train_original = X
        self.y_train_original = y
        self.channels, self.elements_per_channel, self.Input_dims, self.Output_dims = self.determine_dims(X, y)
############################################################################################

############################# Basic model construction #####################################
    #This method builds a mosm model with training data (X,y) and the specified
    #hyperparameters. If the hyperparameters are not specified, then they are
    #chosen randomly upon creation.
    #Remember that the dimensionality of each spectral_element is as follows:
    #spectral_constant -> output_dimensions
    #spectral_mean -> input_dimensions * output_dimensions
    #spectral_variance -> input_dimensions * output_dimensions
    #spectral_delay -> input_dimensions * output_dimensions
    #spectral_phase -> output_dimensions
    #spectral_noise -> output_dimensions
    #So, for example, spectral_means is a list of 2D arrays with shape (input_dim, output_dim).
    #For a model with Q = 3 (3 components), we need to specify 3 multi-output spectral mixture kernels,
    #so we need 3 elements for spectral_constants, each of size output_dimensions.
    def build_model(self, spectral_constants=None, spectral_means=None, spectral_variances=None, spectral_delays=None, spectral_phases=None, spectral_noises=None, likelihood_variance=None):
        args = [spectral_constants, spectral_means, spectral_variances, spectral_delays, spectral_phases, spectral_noises]
        args = [[None]*self.Q if x is None else x for x in args]
        # kernel = MOSM(self.Input_dims, self.Output_dims, spectral_constants[0], spectral_means[0], spectral_variances[0], spectral_delays[0], spectral_phases[0], spectral_noises[0])
        kernel = MOSM(self.Input_dims, self.Output_dims, args[0][0], args[1][0], args[2][0], args[3][0], args[4][0], args[5][0])
        for i in range(1, self.Q):
            kernel+=MOSM(self.Input_dims, self.Output_dims, args[0][i], args[1][i], args[2][i], args[3][i], args[4][i], args[5][i])
        self.model = gpflow.models.GPR(self.X_train_original, self.y_train_original, kernel)
        if(likelihood_variance != None):
            self.model.likelihood.variance = likelihood_variance
        self.make_optimization_tools(self.optimization_method)
        self.is_model_anchored = True

    #optimizer can be any of the following: 'Nelder-Mead', 'Powell', 'CG', 'BFGS','Newton-CG', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov'.
    #However, some of the optimizers need additional parameters (for example, some need a jacobian function). Refer to https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html for more information.
    #For this particular kernel, L-BFGS-B gives stable results.
    # def change_optimizer(self, optimizer):
    #     # self.opt = gpflow.train.ScipyOptimizer(method=optimizer)
    #     self.make_optimization_tools(optimizer)

    def make_optimization_tools(self, optimization_method=None, var_list=None):
        if(optimization_method is None):
            optimization_method = self.optimization_method
        self.opt = gpflow.train.ScipyOptimizer(method=optimization_method)
        self.optimizer_tensor = self.opt.make_optimize_tensor(self.model, gpflow.get_default_session(), var_list=var_list, maxiter=10, disp=True)
    #Refer to https://github.com/GPflow/GPflow/issues/756 for slow down issues.
    # def optimize(self, iters=5000, display=True, anchor=False):
    #     #The following code can be used to decrease re-train time if you want to loop optimization rounds.
    #     #The idea is to refresh the graph in case a new model is used.
    #     # self.opt = gpflow.train.ScipyOptimizer(method=self.opt_method)
    #     # tf.reset_default_graph()
    #     # graph = tf.get_default_graph()
    #     # gpflow.reset_default_session(graph=graph)
    #     self.opt.minimize(self.model, maxiter=iters, disp=display, anchor=anchor)

    def optimize(self, iterations=1000, display=True):
        (self.optimizer_tensor.optimizer_kwargs['options'])['maxiter'] = iterations
        (self.optimizer_tensor.optimizer_kwargs['options'])['disp'] = display
        session = gpflow.get_default_session()
        self.optimizer_tensor.minimize(session=session)
        self.is_model_anchored = False

    def anchor_model(self):
        self.model.anchor(gpflow.get_default_session())
        self.is_model_anchored = True

    def optimize2(self, iters=5000):
        session = gpflow.get_default_session()
        for it in range(iters):
            print("ITERATION ", it)
            session.run(self.optimizer_tensor)

    #Returns the GPFlow model.
    def get_model(self):
        return self.model
############################################################################################

################################ Prediction ################################################
    #Takes an X_prediction in multioutput format and computes the corresponding
    #y_prediction. It should be noted that X_prediction does not need to include
    #data points for all channels, just the desired ones.
    def predict(self, X_prediction):
        self.X_prediction = X_prediction
        #Alternatively, you could predict y.
        # self.prediction_mean, self.prediction_std = self.model.predict_y(X_prediction)
        self.prediction_mean, self.prediction_var = self.model.predict_f(X_prediction)
        self.prediction_channels, self.elements_per_channel_prediction = self.determine_prediction_dims(X_prediction, self.prediction_mean, self.prediction_var)
        return self.prediction_mean, self.prediction_var

    #Finds min or max values for X_channel at every input dimension.
    def find_border_points(self, X_channel, min_or_max):
        input_dims = len(X_channel[0])
        points = []
        function = None
        if(min_or_max == 0):
            function = np.min
        else:
            function = np.max
        for index in range(input_dims):
            points.append(function(np.array(X_channel)[:,index]))
        return points

    #Creates X_predict at given resolution (number of points) for the
    #specified channels at specified intervals. If the starting or ending
    #points of the intervals are not specified then they are taken to be the
    #minimum and maximum elements of the x-axis elements for each channel
    #according to the training set.

    #A valid start or end specification contains values for all channels. If
    #the value for a given channel is not specified, then it must be marked by
    #None, so the method can find it.

    #e.g: If the selected channels are [0,1,2] then a valid start would be
    #[0.3, 0.4, 0.1], where 0.3 would be the starting point of channel 0,
    #0.4 of channel 1 and 0.1 of channel 2.
    #If no start is specified, then all starting values are computed from
    #the training set.

    #Channels can come in any order, and this will determine plotting order.
    #e.g: If channels is [0,2,1], we may specify end = [None, 0.3, None], in
    #which case the starting point for channels 0 and 1 will be computed from
    #the training set, while channel 2 will end at 0.3
    def predict_interval(self, resolution, channels, start=None, end=None):
        def compute_points(given_by_user_bounds, min_or_max):
            bounds = {}
            slot = 0
            if (given_by_user_bounds != None):
                while(slot < len(channels)):
                    if (given_by_user_bounds[slot] != None):
                        bounds[channels[slot]] = given_by_user_bounds[slot]
                    else:
                        bounds[channels[slot]] = self.find_border_points(self.X.get(channels[slot]), min_or_max)
                    slot = slot + 1
            else:
                for channel in channels:
                    bounds[channel] = self.find_border_points(self.X.get(channel), min_or_max)
            return bounds

        starting_points = compute_points(start, 0)
        end_points = compute_points(end, 1)

        X_predict = []
        input_dims = len(self.X.get(channels[0])[0])
        for channel in channels:
            linspaces = [np.linspace(starting_points.get(channel)[i], end_points.get(channel)[i], resolution) for i in range(input_dims)]
            counters = np.zeros(input_dims)
            for index in range(resolution**input_dims): #   We're generating all possible combinations of points.
                point = [channel]
                for dim in range(input_dims):
                    point.append(linspaces[dim][int(counters[dim])])
                    counters[dim] = (counters[dim] + 1)%resolution
                X_predict.append(point)
        return np.array(X_predict)

    #Returns posterior mean and var. Using this function only makes sense after
    #predictions.
    def get_prediction_info(self):
        return self.prediction_mean, self.prediction_var
############################################################################################

################################## Plotting tools ##########################################
    #Latent function must take numpy array as argument and return the function
    #applied to each element of the array. The purpose of specifying latent functions is to
    #plot them against the predictions, in case they are known.
    def define_latent_functions(self, functions, channels):
        self.latent_functions = {}
        for i in range(len(channels)):
            self.latent_functions[channels[i]] = functions[i]

    #We can add extra observations, to each channel, for plotting purposes.
    def add_extra_observations(self, X, y, channels):
        for index in range(len(X)):
            channel = channels[index]
            channel_list = []
            for element in X[index]:
                new_x_element = []
                if(isinstance(element,(list, np.ndarray))):
                    for x_i in element:
                        new_x_element.append(x_i)
                else:
                    new_x_element.append(element)
                new_x_element = np.array(new_x_element)
                channel_list.append(new_x_element)
            self.X_extra[channel] = channel_list #Multi-dim case?
            self.y_extra[channel] = np.array([np.array(yy) for yy in y[index]]).reshape(-1,1)

    #Currently working for 1 feature in x axis (usually considered to be time).
    #If your input dimension is greater than 1 then you're going to have to resort
    #to other tricks to make your plots.
    def make_plots(self, savename, var=True, shape=None, dpi=300, title = "Experiment", show=False):
        if (self.Input_dims > 2):
            print("This is only defined for one or two dimensional inputs")
            return
        if(self.Input_dims == 2):
            self.make_plots_3d(savename, var, shape, dpi)
        sns.set(font_scale=3)
        with sns.axes_style("darkgrid"):
            sns.set_style("whitegrid")
            # sns.set(rc={"font.size":20,"axes.labelsize":20})
            # plt.rcParams.update({'font.size': 22})
        #Uncomment the following 2 and comment the above 2 to change to dark theme
        # with sns.axes_style("dark"):
            # sns.set_style("dark", {"axes.facecolor": ".1"})
            # plt.subplots_adjust(top=0.85)
            if (shape == None):
                shape = [len(self.prediction_channels), 1]

            fig, axes = plt.subplots(shape[0], shape[1], figsize=(shape[1]*10 + 10, shape[0]*5), sharey=False, constrained_layout=True, squeeze=False)
            fig.suptitle(title, fontsize=36)

            legend_elements = [Line2D([0], [0], color='k', lw=3, label='Posterior mean')]
            #We suppose that X has one feature (Input_dim = 1)
            X_pred_one_feature = {}
            X_obs_one_feature = {}
            X_extra_one_feature = {}

            for channel in self.prediction_channels:
                x_pred = self.X_pred.get(channel)
                x_obs = self.X.get(channel)
                X_pred_one_feature[channel] = np.array([y[0] for y in x_pred])
                X_obs_one_feature[channel] = np.array([y[0] for y in x_obs])
                x_extra = self.X_extra.get(channel)
                if(x_extra != None):
                    # x_extra = self.X_extra.get(channel)
                    X_extra_one_feature[channel] = [y[0] for y in x_extra]

            is_latent_available = 0
            are_training_obs_plotted = 0
            for channel in self.prediction_channels:
                prediction_channel_mapping = self.pred_channel_mapping[channel] #This allows us to plot the channels in the specified order of the X_prediction.
                plot_mapping_for_channel_y = int(prediction_channel_mapping % shape[1])
                plot_mapping_for_channel_x = int(prediction_channel_mapping/shape[1])
                latent_function_candidate = self.latent_functions.get(channel) #In case some latent functions have been defined we use them to compute the latent curves for the current resolution given by the X prediction data.
                latent_y = None

                #For each of the subplots we'll only consider the space defined
                #by the prediction X values. All the curves defined within this
                #interval will be considered for the y_min and y_max values,
                #for each subplot.
                x_min = np.min(X_pred_one_feature.get(channel))
                x_max = np.max(X_pred_one_feature.get(channel))
                valid_x_values = np.where(np.logical_and(X_obs_one_feature.get(channel)>=x_min, X_obs_one_feature.get(channel)<=x_max)) #We find the original observations that survive within prediction range.
                X_to_plot = X_obs_one_feature.get(channel)[valid_x_values]
                y_to_plot = self.y.get(channel)[valid_x_values]
                y_min = np.min(self.mean_pred.get(channel)) #We always have predictions, so we begin our search for plot limits here.
                y_max = np.max(self.mean_pred.get(channel))

                #We could compute these at plotting statements, but the verbose approach was chosen for clarity of exposition.
                if(self.X_extra): #If extra observations are available.
                    if(X_extra_one_feature.get(channel)):
                        valid_x_values_extra = np.where(np.logical_and(X_extra_one_feature.get(channel) >=x_min, X_extra_one_feature.get(channel) <=x_max)) #We take only those that fit within the prediction range.
                        X_to_plot_extra_data = np.array(X_extra_one_feature.get(channel))[valid_x_values_extra]
                        y_to_plot_extra_data = self.y_extra.get(channel)[valid_x_values_extra]
                        if(len(X_to_plot_extra_data) > 0): #If any points survive, at all, within the prediction range, then we consider them for the limits of the subplot.
                            y_min = min(np.min(y_to_plot_extra_data), y_min)
                            y_max = max(np.max(y_to_plot_extra_data), y_max)
                if(y_to_plot.shape[0] > 0): #If our original observations are contained within the prediction interval we have to consider them for the limits.
                    y_min = min(np.min(y_to_plot), y_min)
                    y_max = max(np.max(y_to_plot), y_max)
                if(var == True): #If we're plotting variances, then we must consider both curves for the limits.
                    lower_variance_curve = (self.mean_pred.get(channel) - 2 * self.std_pred.get(channel)).reshape([-1])
                    y_min = min(y_min, np.min(lower_variance_curve))
                    upper_variance_curve = (self.mean_pred.get(channel) + 2 * self.std_pred.get(channel)).reshape([-1])
                    y_max = max(y_max, np.max(upper_variance_curve))
                if (latent_function_candidate != None): #If we have a latent function it must also be considered for the limits of the plot.
                    latent_y = latent_function_candidate(X_pred_one_feature.get(channel))
                    is_latent_available = 1
                    y_min = min(np.min(latent_y), y_min)
                    y_max = max(np.max(latent_y), y_max)

                diff_x = abs(x_max - x_min)/100 #These positive values are used to enlarge the limit of the plots by a small margin.
                diff_y = abs(y_max - y_min)/100 #The idea is to avoid chopping points or plot lines.
                current_axis = axes[plot_mapping_for_channel_x, plot_mapping_for_channel_y]

                if(self.X_extra):
                    if(X_extra_one_feature.get(channel)):
                        current_axis.plot(X_to_plot_extra_data, y_to_plot_extra_data, 'b.', ms=20) #Plot the extra observations, within prediction range.
                current_axis.plot(X_pred_one_feature.get(channel), self.mean_pred.get(channel), 'k-', lw=3) #Plot the posterior mean.
                current_axis.plot(X_to_plot, y_to_plot, 'r.', ms=20) #Plot the observations used to train the model, within prediction range.
                if(len(X_to_plot) > 0):
                    are_training_obs_plotted = 1
                if(var == True):
                    current_axis.fill_between(X_pred_one_feature.get(channel), lower_variance_curve, upper_variance_curve, alpha=0.2) #Plot the posterior variance.
                if(latent_function_candidate != None):
                    current_axis.plot(X_pred_one_feature.get(channel), latent_y, 'g--') #Plot the latent function for the prediction points, if it is available.
                current_axis.set_xlim([x_min - diff_x, x_max + diff_x]) #Modify the limits of each subplot.
                current_axis.set_ylim([y_min - diff_y, y_max + diff_y])
            if(are_training_obs_plotted==1):
                legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Training Observations', markerfacecolor='r', markersize=15))
            if(self.X_extra and (len(X_to_plot_extra_data) > 0)):
                legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Removed Observations', markerfacecolor='b', markersize=15))
            if(is_latent_available==1):
                legend_elements.append(Line2D([0], [0], color='g', lw=3, ls='dashed', label='Latent function'))
            plt.legend(handles=legend_elements, loc='best', prop={'size': 16})
            plt.savefig(savename,dpi=100)
            if(show==True):
                plt.show()

    #Coming maybe.
    def make_plots_3d(self, savename, var, shape):
        pass

    def determine_prediction_dims(self, X_pred, mean, var):
        channels, elements_per_channel = np.unique(X_pred[:,0], return_counts=True)
        self.X_pred = {}
        self.mean_pred = {}
        self.var_pred = {}
        self.std_pred = {}
        start = 0
        end = 0
        channels = [int(x) for x in channels]
        self.pred_channel_mapping = {} #This is used to plot using the channel order given by the predictions.
        count = 0
        for index in range(len(channels)):
            channel = channels[index]
            self.pred_channel_mapping[channel] = count #A mapping used to preserve prediction channel order while plotting.
            end = end + elements_per_channel[count]
            self.X_pred[channel] = [x for x in X_pred[:,1:][start:end]]
            self.mean_pred[channel] = mean[start:end]
            self.var_pred[channel] = var[start:end]
            self.std_pred[channel] = np.sqrt(self.var_pred[channel])
            start = end
            count = count + 1
        return channels, elements_per_channel

    #Repackages training data to ease plotting and finds useful information about
    #channels and the number of elements per channel.
    def determine_dims(self, X, y):
        channels, elements_per_channel = np.unique(X[:,0], return_counts=True)
        self.X = {}
        self.y = {}
        start = 0
        end = 0
        channels = [int(x) for x in channels]
        for channel in channels:
            end = end + elements_per_channel[channel]
            self.X[channel] = [x for x in X[:,1:][start:end]]
            self.y[channel] = y[start:end]
            start = end
        return channels, elements_per_channel, (X[0].shape[0] - 1), len(channels)
############################################################################################

##################### Remove intervals from training set ###################################
    #Creates random starting and ending points that define intervals to be removed
    #from the original signal.
    def make_random_deleted_intervals(self, number_of_starting_points):
        starting_points = np.sort(np.random.rand(number_of_starting_points))
        deltas = np.random.rand(number_of_starting_points)/10
        ending_limits = list(starting_points[1:])
        ending_limits.append(1.0)

        bounded_ends = []
        prev = 0
        for index, value in enumerate(starting_points):
            new_end_point = min(starting_points[index]+deltas[index], ending_limits[index])
            bounded_ends.append(new_end_point)

        bound_list = []
        prev_end = -1
        for start,end in zip(starting_points, bounded_ends):
            if (prev_end == start):
                bound_list[len(bound_list)-1] = end
            else:
                bound_list.append(start)
                bound_list.append(end)
            prev_end = end
        reshaped_list = np.array(bound_list).reshape(-1,2)
        return reshaped_list[:,0], reshaped_list[:,1]

    #Transforms the starting and ending points of deletion intervals into a selection
    #array, which indicates whether a data point should be kept or removed.
    def compute_selection_list(self, num_elements, starts, ends):
        selection_list = np.ones(num_elements, dtype=bool)
        for start, end in zip(starts, ends):
            selection_list[int(start*num_elements):int(end*num_elements)] = False
        return selection_list

    #Makes sure that there is, at least, one sample point in one channel for each time.
    def enforce_one_sample_per_time(self, selection_lists):
        for idx, element in enumerate(selection_lists[0]):
            is_included = False
            for list in selection_lists:
                is_included = is_included or list[idx]
            if(not is_included):
                chosen_list = random.choice(selection_lists)
                chosen_list[idx] = True
        return selection_lists

    #Filters elements in data according to selection, which is a boolean array
    #that indicates whether a data point is to be kept or not.
    def select_elements(self, data, selection):
        selected_data = []
        removed_data = []
        if isinstance(data,np.ndarray):
            tuples = zip(np.nditer(data), selection)
        else:
            tuples = zip(data, selection)

        for element, criteria in tuples:
            if criteria:
                selected_data.append(element)
            else:
                removed_data.append(element)
        return np.array(selected_data), np.array(removed_data)

    #Makes 5 random deletion intervals and computes the selection list, for each channel.
    #After that the data points are filtered and the separate lists are returned.
    def remove_slabs(self, X, y):
        X_removed = [[] for channel in range(len(X))]
        y_removed = [[] for channel in range(len(X))]
        X_kept = [[] for channel in range(len(X))]
        y_kept = [[] for channel in range(len(X))]
        selection_lists = []
        for channel in range(len(X)):
            starts_for_channel, ends_for_channel = self.make_random_deleted_intervals(5)
            selection_list = self.compute_selection_list(len(X[channel]), starts_for_channel, ends_for_channel)
            selection_lists.append(selection_list)

        selection_lists = self.enforce_one_sample_per_time(selection_lists)

        for channel in range(len(X)):
            X_kept[channel], X_removed[channel] = self.select_elements(X[channel], selection_lists[channel])
            y_kept[channel], y_removed[channel] = self.select_elements(y[channel], selection_lists[channel])
        return X_kept, y_kept, X_removed, y_removed

    #This function allows us to remove elements from a given channel, specifying
    #the endpoints of the deletion interval in terms of percentages.
    def modify_channel(self, X, y, X_removed, y_removed, channel, start_percent, end_percent):
        if(start_percent < 0 or end_percent > 1 or end_percent < start_percent):
            print("Not a valid starting point")
            return X,y
        X[channel] = np.array(X[channel])
        starting_point = int((X[channel].shape[0]*start_percent))
        end = int((X[channel].shape[0]*end_percent))

        X_before = X[channel][0:starting_point]
        y_before = y[channel][0:starting_point]
        X_modified = X[channel][starting_point:end]
        y_modified = y[channel][starting_point:end]
        X_after = X[channel][end:X[channel].shape[0]]
        y_after = y[channel][end:X[channel].shape[0]]
        X[channel] = X_modified
        y[channel] = y_modified

        # X_removed = []
        for element in X_before:
            X_removed[channel].append(element)
        for element in X_after:
            X_removed[channel].append(element)

        # y_removed = []
        for element in y_before:
            y_removed[channel].append(element)
        for element in y_after:
            y_removed[channel].append(element)

        return X, y, X_removed, y_removed
############################################################################################

################################ Restrict parameters #######################################
    #Returns the corresponding model parameter, for kernel q.
    def get_referenced_parameter_multikern(self, parameter, q):
        if(parameter == 'constants'):
            return self.model.kern.kernels[q].constant
        if(parameter == 'variances'):
            return self.model.kern.kernels[q].variance
        if(parameter == 'phases'):
            return self.model.kern.kernels[q].phase
        if(parameter == 'delays'):
            return self.model.kern.kernels[q].delay
        if(parameter == 'means'):
            return self.model.kern.kernels[q].mean
        if(parameter == 'noises'):
            return self.model.kern.kernels[q].noise

    #Same as above, but note that when there is only one kernel
    #in the model then the way to access the parameters changes.
    def get_referenced_parameter(self, parameter):
        if(parameter == 'constants'):
            return self.model.kern.constant
        if(parameter == 'variances'):
            return self.model.kern.variance
        if(parameter == 'phases'):
            return self.model.kern.phase
        if(parameter == 'delays'):
            return self.model.kern.delay
        if(parameter == 'means'):
            return self.model.kern.mean
        if(parameter == 'noises'):
            return self.model.kern.noise

    #Changes whether the specified parameter can be trained or not.
    #Useful for cascading heuristics.
    def change_trainability(self, parameter, ToF):
        if (self.Q == 1):
            self.get_referenced_parameter(parameter).trainable = ToF
        else:
            for q in range(self.Q):
                self.get_referenced_parameter_multikern(parameter,q).trainable = ToF

    #Makes the parameters in train list trainable and the parameters in fix list
    #untrainable. Note that the lists do not contain the parameters themselves,
    #but just strings with their names.
    def toggle_trainables(self, train, fix):
        for parameter in train:
            self.change_trainability(parameter, True)
        for parameter in fix:
            self.change_trainability(parameter, False)
############################################################################################

################################ Save and Load #############################################
    #Creates a dictionary containing all the model hyperparameters, mainly so they can be
    #saved to disk. The original GPFlow names are preserved in the dictionary. This is done
    #to avoid having to resort to model.anchor(session), due to the massive time cost.
    #To learn more about this: https://gpflow.readthedocs.io/en/develop/notebooks/tips_and_tricks.html
    def bundle_hyperparameters(self):
        bundle = {}
        session = gpflow.get_default_session()
        if(self.Q > 1):
            for q in range(self.Q):
                bundle['GPR/kern/kernels/%d/constant' % q] = self.model.kern.kernels[q].constant.read_value(session)
                bundle['GPR/kern/kernels/%d/delay' % q] = self.model.kern.kernels[q].delay.read_value(session)
                bundle['GPR/kern/kernels/%d/mean' % q] = self.model.kern.kernels[q].mean.read_value(session)
                bundle['GPR/kern/kernels/%d/phase' % q] = self.model.kern.kernels[q].phase.read_value(session)
                bundle['GPR/kern/kernels/%d/variance' % q] = self.model.kern.kernels[q].variance.read_value(session)
        elif(self.Q == 1):
            bundle['GPR/kern/constant'] = self.model.kern.constant.read_value(session)
            bundle['GPR/kern/delay'] = self.model.kern.delay.read_value(session)
            bundle['GPR/kern/mean'] = self.model.kern.mean.read_value(session)
            bundle['GPR/kern/phase'] = self.model.kern.phase.read_value(session)
            bundle['GPR/kern/variance'] = self.model.kern.variance.read_value(session)
        bundle['GPR/likelihood/variance'] = self.model.likelihood.variance.read_value(session)
        return bundle

    def anchor_fast(self):
        session = gpflow.get_default_session()
        if(self.Q > 1):
            for q in range(self.Q):
                self.model.kern.kernels[q].constant.value = self.model.kern.kernels[q].constant.read_value(session)
                self.model.kern.kernels[q].delay.value = self.model.kern.kernels[q].delay.read_value(session)
                self.model.kern.kernels[q].mean.value = self.model.kern.kernels[q].mean.read_value(session)
                self.model.kern.kernels[q].phase.value = self.model.kern.kernels[q].phase.read_value(session)
                self.model.kern.kernels[q].variance.value = self.model.kern.kernels[q].variance.read_value(session)
        elif(self.Q == 1):
            self.model.kern.constant.value = self.model.kern.constant.read_value(session)
            self.model.kern.delay.value = self.model.kern.delay.read_value(session)
            self.model.kern.mean.value = self.model.kern.mean.read_value(session)
            self.model.kern.phase.value = self.model.kern.phase.read_value(session)
            self.model.kern.variance.value = self.model.kern.variance.read_value(session)
        self.model.likelihood.variance.value = self.model.likelihood.variance.read_value(session)
        self.is_model_anchored = True

    #Saves the model parameters as a dictionary. Also saves the original training data given to the model.
    def save(self, name):
        if(self.is_model_anchored):
            self.toggle_trainables(['delays', 'variances', 'constants', 'means', 'phases', 'noises'],[])
            hyperparameters = self.read_trainables()
        else:
            hyperparameters = self.bundle_hyperparameters()
        np.save("parameters_" + name, hyperparameters)
        np.save("train_data_" + name, {'X' : self.X_train_original, 'y': self.y_train_original})

    #For Q > 1, returns the model parameters of kernel component q, from the dictionary 'parameters'.
    def get_spectral_parameters_for_one_mosm_kernel_component_by_q(self, parameters, q):
        return parameters['GPR/kern/kernels/%d/constant' % q], parameters['GPR/kern/kernels/%d/delay' % q], parameters['GPR/kern/kernels/%d/mean' % q], parameters['GPR/kern/kernels/%d/noise' % q], parameters['GPR/kern/kernels/%d/phase' % q], parameters['GPR/kern/kernels/%d/variance' % q]

    #Returns the parameters of kernel component q, from the dictionary 'parameters'.
    def get_spectral_parameters_for_one_mosm_kernel_component(self, parameters, q):
        if (q == 0):
            if (True in [True if 'kern/kernels/' in x else False for x in parameters.keys()]):
                return self.get_spectral_parameters_for_one_mosm_kernel_component_by_q(parameters, q)
            else:
                return parameters['GPR/kern/constant'], parameters['GPR/kern/delay'], parameters['GPR/kern/mean'], parameters['GPR/kern/noise'], parameters['GPR/kern/phase'], parameters['GPR/kern/variance']
        else:
            return self.get_spectral_parameters_for_one_mosm_kernel_component_by_q(parameters, q)

    #Remakes the model from the saved parameters, which were stored as a dictionary.
    # def build_model_from_saved_parameters(self, X, y, parameters):
    #     self.add_training_data(X,y)
    #     constants, delays, means, noises, phases, variances = self.get_spectral_parameters_for_one_mosm_kernel_component(parameters, 0)
    #     kernel = MOSM(self.Input_dims, self.Output_dims, spectral_constant = constants, spectral_mean = means, spectral_variance = variances, spectral_delay = delays, spectral_phase = phases, spectral_noise = noises)
    #     for i in range(1,self.Q):
    #         constants, delays, means, noises, phases, variances = self.get_spectral_parameters_for_one_mosm_kernel_component(parameters, i)
    #         kernel += MOSM(self.Input_dims, self.Output_dims, spectral_constant = constants, spectral_mean = means, spectral_variance = variances, spectral_delay = delays, spectral_phase = phases, spectral_noise = noises)
    #     self.model = gpflow.models.GPR(X,y,kernel)
    #     self.model.likelihood.variance = parameters['GPR/likelihood/variance']
    #     self.make_optimization_tools()
    #     self.is_model_anchored = True

    def build_model_from_saved_parameters(self, X, y, parameters):
        self.add_training_data(X,y)
        spectral_constants = []
        spectral_delays = []
        spectral_means = []
        spectral_noises = []
        spectral_phases = []
        spectral_variances = []
        for i in range(self.Q):
            constants, delays, means, noises, phases, variances = self.get_spectral_parameters_for_one_mosm_kernel_component(parameters, i)
            spectral_constants.append(constants)
            spectral_delays.append(delays)
            spectral_means.append(means)
            spectral_noises.append(noises)
            spectral_phases.append(phases)
            spectral_variances.append(variances)
        self.build_model(spectral_constants=spectral_constants, spectral_means=spectral_means, spectral_variances=spectral_variances, spectral_delays=spectral_delays, spectral_phases=spectral_phases, spectral_noises=spectral_noises, likelihood_variance=parameters['GPR/likelihood/variance'])

    #Loads the model parameters from disk.
    def load(self, name):
        parameters = np.load("parameters_" + name +".npy").item()
        train_data = np.load("train_data_" + name +".npy").item()
        self.build_model_from_saved_parameters(train_data['X'], train_data['y'], parameters)

    #Prints the model parameters as a pandas table. Cannot be used before fit,
    #since GPFLow model has not been created before that point.
    def as_pandas_table(self):
        return self.model.as_pandas_table()

    #Returns a dictionary with all currently trainable parameters of the model.
    def read_trainables(self):
        return self.model.read_trainables()
############################################################################################

###################### Spectral estimation of most important frequencies ###################
    #https://arxiv.org/pdf/1703.09824.pdf page 17 could be a better method.
    def nyquist_estimation(self, train_x):
        input_dim = np.shape(train_x)[1]

        if np.size(train_x.shape) == 1:
            train_x = np.expand_dims(train_x ,-1)

        if np.size(train_x.shape) == 2:
            train_x = np.expand_dims(train_x ,0)

        train_x_sort = np.copy(train_x)
        train_x_sort.sort(axis=1)

        min_dist_sort = np.squeeze(np.abs(train_x_sort[: ,1:, :] - train_x_sort[: ,:-1, :]))
        min_dist = np.zeros([input_dim] ,dtype=float)

        for ind in np.arange(input_dim):
            try:
                min_dist[ind] = min_dist_sort[np.amin(np.where(min_dist_sort[:,ind] > 0), axis=1), ind]
            except:
                min_dist[ind] = min_dist_sort[np.amin(np.where(min_dist_sort > 0), axis=1)]

        nyquist = np.divide(0.5,min_dist)
        return nyquist

    #Computes bayesian nonparametric spectral estimation (https://papers.nips.cc/paper/8216-bayesian-nonparametric-spectral-estimation.pdf)
    #for a given signal and then selects the frequency peaks (those that concentrate the most power of the signal).
    def compute_bnse(self, X, y):
        nyq = self.nyquist_estimation(X)
        my_bse = bse(X, y)
        my_bse.set_freqspace(nyq)
        my_bse.train()
        my_bse.compute_moments()
        peaks, amplitude = my_bse.get_freq_peaks()
        return peaks, amplitude, my_bse

    #Computes the Q most relevant frequencies of a signal. If there are q < Q candidates, then the
    #candidates are repeated, with jitter, until Q candidates are reached.
    def compute_freq_peaks_bnse(self, X, y, Q):
        peaks, amplitude, bse = self.compute_bnse(X,y)
        sorted_info = [[x,y] for x,y in sorted(zip(amplitude,peaks), key=lambda pair: pair[0])][-Q:]
        most_significant_peaks = [x[1] for x in sorted_info]
        most_significant_peaks_amplitudes = [x[0] for x in sorted_info]
        if (len(most_significant_peaks) == 0): #In case no peaks were found.
            return np.random.rand(Q)*2*np.pi, np.ones(Q)/Q
        slot = len(most_significant_peaks)-1
        while len(most_significant_peaks) < Q:
            most_significant_peaks.append(most_significant_peaks[slot] + (np.random.standard_t(3,1)*0.01)[0])
            most_significant_peaks_amplitudes.append(most_significant_peaks_amplitudes[slot])
            slot = (slot-1)%len(most_significant_peaks)
        return 2*np.pi*np.array(most_significant_peaks), np.array(most_significant_peaks_amplitudes)

    #Computes the most relevant frequencies of the training data signals, in order to be used as the starting
    #point for MOSM optimisation.
    def get_starting_freq(self):
        freqs = []
        for channel in self.channels:
            channel_X = np.array(self.X[channel])
            channel_y = np.array(self.y[channel])
            peaks, amplitudes = self.compute_freq_peaks_bnse(channel_X, channel_y, self.Q)
            freqs.append(peaks)

        spectral_means = np.transpose(np.array(freqs).reshape((len(self.channels),self.Q)))
        return [[x] for x in spectral_means]
############################################################################################

############################# Single channel Spectral Mixture ##############################
    #Computes the single channel spectral mixture GP for a given channel. Note that the
    #single channel spectral mixture used is given by the case i=j of the multi-output
    #spectral mixture kernel.
    def compute_spectral_mixture_for_a_single_channel(self, channel):
        iterations = 10
        channel_X = np.array(self.X[channel])
        channel_y = np.array(self.y[channel])

        means_guess_orig, amplitudes = self.compute_freq_peaks_bnse(channel_X, channel_y, self.Q)
        weights_orig = amplitudes*channel_y.std()
        weights_orig = weights_orig/np.sum(weights_orig)
        weights_orig = np.sqrt(weights_orig)

        hp_candidate = None
        nll_value = None
        not_first = 0
        for it in range(iterations):
            try:
                tf.reset_default_graph()
                graph = tf.get_default_graph()
                gpflow.reset_default_session(graph=graph)
                means_guess = means_guess_orig
                weights = weights_orig
                scales = np.random.random((1, self.Q))
                if(not_first == 1):
                    means_guess = means_guess + np.random.standard_t(3,means_guess.shape)*0.1
                    weights = weights_orig + np.random.standard_t(3,weights_orig.shape)*0.1
                else:
                    not_first = 1
                means = means_guess.reshape(-1,1)
                scales = scales.reshape(-1,1)
                k_lin = gpflow.kernels.Linear(self.Input_dims)
                k_sm = SpectralMixture(num_mixtures= self.Q, mixture_weights=weights, mixture_scales=scales, mixture_means=means, input_dim=self.Input_dims)
                k_sm_sum=k_sm+k_lin
                m_sm = gpflow.models.GPR(channel_X, channel_y, kern=k_sm_sum)
                m_sm.kern.kernels[0].mixture_means.trainable = False
                gpflow.train.ScipyOptimizer(method='BFGS').minimize(m_sm, maxiter=1000, disp=True, anchor=True)
                m_sm.kern.kernels[0].mixture_means.trainable = True
                gpflow.train.ScipyOptimizer(method='BFGS').minimize(m_sm, maxiter=1000, disp=True, anchor=True)
                nll = -m_sm.compute_log_likelihood()
                if(nll_value == None):
                    nll_value = nll
                    hp_candidate = self.transform_trainables_into_hp_vector(m_sm.read_trainables())
                elif(nll < nll_value):
                    nll_value = nll
                    hp_candidate = self.transform_trainables_into_hp_vector(m_sm.read_trainables())
            except:
                pass
        return hp_candidate[:-2], means_guess_orig

    #Computes the single channel spectral mixture GP for a given channel, using bayesian optimisation
    #to find the starting hyperparameters that produce the best model.
    def compute_spectral_mixture_for_a_single_channel_bayesianly(self, channel):
        def plot(m):
            xx = np.linspace(0, 5, 300)[:,None]
            mean, var = m.predict_f(xx)
            plt.figure(figsize=(24, 12))
            plt.plot(channel_X, channel_y, 'kx', mew=2)
            plt.plot(xx, mean, 'b', lw=2)
            plt.fill_between(xx[:,0], mean[:,0] - 2*np.sqrt(var[:,0]), mean[:,0] + 2*np.sqrt(var[:,0]), color='blue', alpha=0.2)
            plt.show()

        iterations = 10
        channel_X = np.array(self.X[channel])
        channel_y = np.array(self.y[channel])

        means_guess_orig, amplitudes = self.compute_freq_peaks_bnse(channel_X, channel_y, self.Q)

        hp_candidate = []
        nll_value = []
        def objective(hp):
            try:
                #We use the memory saving trick once again.
                tf.reset_default_graph()
                graph = tf.get_default_graph()
                gpflow.reset_default_session(graph=graph)
                weights, means, scales = sm_unravel_hp(hp, self.Q)
                k_sm = SpectralMixture(num_mixtures= self.Q, mixture_weights=weights, mixture_scales=scales, mixture_means=means, input_dim=self.Input_dims)
                k_sm = k_sm + gpflow.kernels.Linear(self.Input_dims)
                m_sm = gpflow.models.GPR(channel_X, channel_y, kern=k_sm)
                m_sm.likelihood.variance = hp[len(hp)-1]
                gpflow.train.ScipyOptimizer(method='BFGS').minimize(m_sm, maxiter=1000, disp=True, anchor=True)
                nll = -m_sm.compute_log_likelihood()

                hp_candidate.append(self.transform_trainables_into_hp_vector(m_sm.read_trainables()))
                nll_value.append(nll)
                plot(m_sm)
                return nll
            except:
                return 10000

        freq_space = [Real(0, 100) for x in range(self.Q)] #weights
        for q in range(2*self.Q):
            freq_space.append(Real(0, 10**2)) #scales and means
        freq_space.append(Real(0, 10**2)) #likelihood_variance
        # freq_space = [Real(10**-5, 10**5) for x in range(2*Q)]

        res_gp = gp_minimize(objective, freq_space, n_calls=30, random_state=0, verbose = True, noise=0.0000001)

        counter = 0
        min_nll = nll_value[counter]
        min_hp = hp_candidate[counter]
        for i in range(len(nll_value)):
            if nll_value[counter] < min_nll:
                min_nll = nll_value[counter]
                min_hp = hp_candidate[counter]
            counter = counter + 1

        return min_hp[:-2], means_guess_orig
############################################################################################

########### Model construction from frequencies and Spectral Mixture parameters ############
    #This method takes the hyperparameters from the single channel spectral mixture GPs and
    #packages them for use in the multi-output spectral mixture version.
    def package_parameters_in_spectral_form(self, parameters_per_channel):
        spectral_means = []
        spectral_variances = []
        spectral_constants = []
        for channel in self.channels:
            for mean in parameters_per_channel[channel][0:self.Q]:
                spectral_means.append(mean)
            for variance in parameters_per_channel[channel][self.Q:2*self.Q]:
                spectral_variances.append(variance)
            for constant in parameters_per_channel[channel][2*self.Q:3*self.Q]:
                spectral_constants.append(constant)
        spectral_means = np.transpose(np.array(spectral_means).reshape((len(self.channels),self.Q)))
        spectral_variances = np.transpose(np.array(spectral_variances).reshape((len(self.channels),self.Q)))
        spectral_constants = np.transpose(np.array(spectral_constants).reshape((len(self.channels),self.Q)))
        return spectral_means, spectral_variances, spectral_constants

    # def build_model_from_freqs(self, spectral_means):
    #     #In case you want to build several models in the same session.
    #     # tf.reset_default_graph()
    #     # graph = tf.get_default_graph()
    #     # gpflow.reset_default_session(graph=graph)
    #     kernel = MOSM(self.Input_dims, self.Output_dims, spectral_mean = [spectral_means[0]])
    #     for i in range(1, self.Q):
    #         kernel+=MOSM(self.Input_dims, self.Output_dims, spectral_mean = [spectral_means[i]])
    #     self.model = gpflow.models.GPR(self.X_train_original, self.y_train_original, kernel)

    #Trains a single-channel spectral mixture kernel GP for each channel and builds a mosm model
    #with the single-channel GPs hyperparameters. Phases and delays are set to random values.
    def compute_starting_parameters(self):
        parameters_per_channel = {}
        for channel in self.channels:
            parameters_per_channel[channel], means_guess = self.compute_spectral_mixture_for_a_single_channel(channel)
        spectral_means, spectral_variances, spectral_constants = self.package_parameters_in_spectral_form(parameters_per_channel)
        self.build_model(spectral_constants=spectral_constants, spectral_means=[spectral_means], spectral_variances=[spectral_variances])

    #Trains a single-channel spectral mixture kernel GP for each channel, using bayesian optimization to find starting hyperparameters.
    #Hyperparameters from the single-channel GPs are used to build a mosm model. Phases and delays are set to random values.
    def compute_starting_parameters_bayesianly(self):
        parameters_per_channel = {}
        for channel in self.channels:
            parameters_per_channel[channel], means_guess = self.compute_spectral_mixture_for_a_single_channel_bayesianly(channel)
        spectral_means, spectral_variances, spectral_constants = self.package_parameters_in_spectral_form(parameters_per_channel)
        self.build_model(spectral_constants=spectral_constants, spectral_means=[spectral_means], spectral_variances=[spectral_variances])

    def compute_starting_parameters_keep_means(self):
        parameters_per_channel = {}
        means_guess_per_channel = []
        for channel in self.channels:
            parameters_per_channel[channel], means_guess = self.compute_spectral_mixture_for_a_single_channel(channel)
            means_guess_per_channel.append(means_guess)
        spectral_means, spectral_variances, spectral_constants = self.package_parameters_in_spectral_form(parameters_per_channel)
        spectral_means = np.transpose(np.array(means_guess_per_channel).reshape((len(self.channels),self.Q)))
        self.build_model(spectral_constants=spectral_constants, spectral_means=spectral_means, spectral_variances=spectral_variances)

    def compute_starting_parameters_keep_means_bayesianly(self):
        parameters_per_channel = {}
        means_guess_per_channel = []
        for channel in self.channels:
            parameters_per_channel[channel], means_guess = self.compute_spectral_mixture_for_a_single_channel_bayesianly(channel)
            means_guess_per_channel.append(means_guess)
        spectral_means, spectral_variances, spectral_constants = self.package_parameters_in_spectral_form(parameters_per_channel)
        spectral_means = np.transpose(np.array(means_guess_per_channel).reshape((len(self.channels),self.Q)))
        self.build_model(spectral_constants=spectral_constants, spectral_means=spectral_means, spectral_variances=spectral_variances)
############################################################################################

############################ Optimisation heuristics #######################################
    #Computes the most relevant frequencies of the training data and uses them as a starting point
    #to optimise the model.
    def optimization_heuristic_zero(self, iterations=1000):
        self.build_model(spectral_means=self.get_starting_freq())
        # self.build_model_from_freqs(self.get_starting_freq())
        self.optimize(iterations = iterations, display=True)

    #Computes the most relevant frequencies of the training data and uses them as a starting point
    #to optimise the model, but optimises the model in cascading fashion.
    #The first step optimises variances and constants, while means, phases and delays remain static.
    #The second step optimises delays, phases and means, while keeping variances and constants static.
    #The third step optimises delays, variances and constants, while keeping means and phases static.
    #The final step optimises all the parameters.
    def optimization_heuristic_one(self, iterations=1000):
        # self.build_model_from_freqs(self.get_starting_freq())
        self.build_model(spectral_means=self.get_starting_freq())
        self.toggle_trainables(['variances', 'constants'],['means', 'phases', 'delays'])
        self.make_optimization_tools()
        self.optimize(iterations = iterations/5, display=True)

        self.toggle_trainables(['delays', 'phases', 'means'],['variances', 'constants'])
        self.make_optimization_tools()
        self.optimize(iterations = iterations/5, display=True)

        self.toggle_trainables(['delays', 'variances', 'constants'],['means', 'phases'])
        self.make_optimization_tools()
        self.optimize(iterations = iterations/5, display=True)

        self.toggle_trainables(['delays', 'variances', 'constants', 'means', 'phases'],[])
        self.make_optimization_tools()
        self.optimize(iterations = iterations*(2/5), display=True)

    #Fits single-channel GPs with spectral mixture kernels to each of the signal channels, in
    #order to use the parameters of these single-channel GPs as a starting point to optimise
    #the mosm model.
    def optimization_heuristic_two(self):
        self.compute_starting_parameters()
        self.optimize(iters = self.iterations, display=True)

    #Fits single-channel GPs with spectral mixture kernels to each of the signal channels,
    #in order to use the parameters of these single-channel GPs as a starting point to
    #begin optimisation of the mosm model. Note that the starting parameters of these
    #single channel GPs are chosen through bayesian optimisation.
    def optimization_heuristic_three(self):
        self.compute_starting_parameters_bayesianly()
        self.optimize(iters=self.iterations, display=True)
############################################################################################

##################################### Metrics ##############################################
    #Given a list of m lists of y values (1 per channel, with m channels) it creates a single list
    #where each element is a list of m values.
    def zip_y_values_for_all_channels(self, original_y_values):
        return [[original_y_values[i][j] for i in range(len(original_y_values))] for j in range(len(original_y_values[0]))]

    #Given a prediction mean and some query points (X_pred) separates the mean predictions according
    #to channel and returns as list.
    def get_means_per_channel_list(self, X_pred, mean):
        channels, elements_per_channel = np.unique(X_pred[:,0], return_counts=True)
        if(isinstance(mean, np.ndarray)):
            working_mean = mean.flatten()
        else:
            working_mean = mean

        means_per_channel = []

        start = 0
        for end in list(accumulate(elements_per_channel)):
            means_per_channel.append([x for x in working_mean[start:end]])
            start = end
        return means_per_channel

    #Computes the mean absolute error for the signals defined by X_values and y_values.
    def compute_mae(self, X_values, y_values):
        X_pred, y_pred = self.transform_lists_into_multioutput_format(X_values, y_values)
        y_true = self.zip_y_values_for_all_channels(y_values)
        prediction_mean, prediction_var = self.model.predict_f(X_pred)
        means_per_channel = self.get_means_per_channel_list(X_pred, prediction_mean)
        y_pred = self.zip_y_values_for_all_channels(means_per_channel)
        return mean_absolute_error(y_true, y_pred)

    #Returns the log-likelihood of the current model.
    def compute_log_likelihood(self):
        return self.model.compute_log_likelihood()
############################################################################################

################################### Old methods ############################################
    #Not defined for q = 1
    #Changes parameter values by a small number.
    def jitter_parameters(self):
        for q in range(self.Q):
            self.model.kern.kernels[q].constant = self.model.kern.kernels[q].constant.value + np.random.standard_t(3, self.model.kern.kernels[q].constant.value.shape)*0.03
            self.model.kern.kernels[q].delay = self.model.kern.kernels[q].delay.value + np.random.standard_t(3, self.model.kern.kernels[q].delay.value.shape)*0.03
            self.model.kern.kernels[q].mean = self.model.kern.kernels[q].mean.value + np.random.standard_t(3, self.model.kern.kernels[q].mean.value.shape)*0.03
            self.model.kern.kernels[q].phase = self.model.kern.kernels[q].phase.value + np.random.standard_t(3, self.model.kern.kernels[q].phase.value.shape)*0.03
            self.model.kern.kernels[q].variance = self.model.kern.kernels[q].variance.value + np.random.standard_t(3, self.model.kern.kernels[q].variance.value.shape)*0.03

    #Not defined for q = 1
    def change_hyperparameters(self, hyperparameters):
        for q in range(self.Q):
            spectral_constant_q, spectral_mean_q, spectral_variance_q, spectral_delay_q, spectral_phase_q = self.get_spectral_parameters(hyperparameters[:len(hyperparameters)-1], q, self.Output_dims, self.Input_dims)
            self.model.kern.kernels[q].constant = spectral_constant_q
            self.model.kern.kernels[q].delay = spectral_delay_q
            self.model.kern.kernels[q].mean = spectral_mean_q
            self.model.kern.kernels[q].phase = spectral_phase_q
            self.model.kern.kernels[q].variance = spectral_variance_q
        self.model.likelihood.variance = hyperparameters[len(hyperparameters)-1]

    def get_spectral_parameters(self, list_of_parameters, index, output_dim, input_dim):
        offset = index*(2*output_dim + 3*input_dim*output_dim)
        spectral_constant_i = np.array(list_of_parameters[offset:offset + output_dim])
        spectral_delay_i = np.array(list_of_parameters[offset + output_dim:offset + output_dim + input_dim*output_dim]).reshape((input_dim, output_dim))
        spectral_mean_i = np.array(list_of_parameters[offset + output_dim + input_dim*output_dim:offset + output_dim + 2*input_dim*output_dim]).reshape((input_dim, output_dim))
        spectral_phase_i = np.array(list_of_parameters[offset + output_dim + 2*input_dim*output_dim:offset + 2*output_dim + 2*input_dim*output_dim])
        spectral_variance_i = np.array(list_of_parameters[offset + 2*output_dim + 2*input_dim*output_dim:offset + 2*output_dim + 3*input_dim*output_dim]).reshape((input_dim, output_dim))
        return spectral_constant_i, spectral_mean_i, spectral_variance_i, spectral_delay_i, spectral_phase_i

    def transform_trainables_into_hp_vector(self, trainables):
        hyperparameters = []
        for element in trainables:
            for items in trainables[element].flatten():
                hyperparameters.append(items)
        return hyperparameters

    # def fit(self, X, y, iters=5000, display=False, hyperparameters = None, anchor=True):
    #     self.build_model(X,y,hyperparameters)
    #     #anchor parameter can be set as False according to the following workaround
    #     #to avoid an exponential increase in re-training time in case we want to
    #     #loop the training of several models in a single session. We will do this
    #     #when we use bayesian optimization to search the starting point of hyperparameters.
    #     #https://github.com/GPflow/GPflow/issues/798
    #     #Keep in mind that if you want your model to persist (e.g: saving it to disk)
    #     #you must fit with anchor = True.
    #     self.opt.minimize(self.model, maxiter=iters, disp=display, anchor=anchor)

    # def build_model(self, X, y, hyperparameters = {}, likelihood_variance=None):
    #     self.add_training_data(X,y)
    #     if(hyperparameters):
    #         kernel = MOSM(self.Input_dims, self.Output_dims)
    #         for i in range(self.Q-1):
    #             kernel+=MOSM(self.Input_dims, self.Output_dims)
    #     else:
    #         spectral_constant_i, spectral_mean_i, spectral_variance_i, spectral_delay_i, spectral_phase_i = self.get_spectral_parameters(hyperparameters, 0, self.Output_dims, self.Input_dims)
    #         kernel = MOSM(self.Input_dims, self.Output_dims, spectral_constant_i, spectral_mean_i, spectral_variance_i, spectral_delay_i, spectral_phase_i)
    #         for i in range(1, self.Q):
    #             spectral_constant_i, spectral_mean_i, spectral_variance_i, spectral_delay_i, spectral_phase_i = self.get_spectral_parameters(hyperparameters, i, self.Output_dims, self.Input_dims)
    #             kernel+=MOSM(self.Input_dims, self.Output_dims, spectral_constant_i, spectral_mean_i, spectral_variance_i, spectral_delay_i, spectral_phase_i)
    #
    #     self.model = gpflow.models.GPR(X, y, kernel)
    #     if(likelihood_variance != None):
    #         self.model.likelihood.variance = likelihood_variance
############################################################################################
