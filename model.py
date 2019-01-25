import gpflow
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from mosm.multi_spectralmixture import MultiSpectralMixture as MOSM
from collections import Counter
import scipy.stats
import random

""" optimizer can be any of the following: 'Nelder-Mead', 'Powell', 'CG', 'BFGS','Newton-CG', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov' """
""" However, some of the optimizers need additional parameters (for example, some need a jacobian function). Refer to https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html for more information """
""" For this particular kernel, L-BFGS-B gives stable results. BFGS gives similar results at a fraction of the computation cost, but is not as stable: It fails to optimize properly roughly 30% of the time. """
""" You could still, however, train several models at the same time cost and reject those that perform horribly bad, just an idea. """
class mosm_model:
    #At model construction we define the number of components and the optimizer
    def __init__(self, number_of_components_q, optimizer = 'L-BFGS-B'):
        self.Q = number_of_components_q
        self.X_extra = {}
        self.y_extra = {}
        self.opt = gpflow.train.ScipyOptimizer(method=optimizer)
        self.latent_functions = {}

    #Takes an X,y pair in the expected multi-output representation, which is as
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
    #Note that observations must be ordered by the x-axis.
    def fit(self, X, y, iters=5000, display=False):
        self.channels, self.elements_per_channel, self.Input_dims, self.Output_dims = self.determine_dims(X, y)
        kernel = MOSM(self.Input_dims, self.Output_dims)
        for i in range(self.Q-1):
            kernel+=MOSM(self.Input_dims, self.Output_dims)
        self.model = gpflow.models.GPR(X, y, kernel)
        self.as_pandas_table()
        self.opt.minimize(self.model, maxiter=iters, disp=display)

    #Latent function must take numpy array as argument and return the function
    #applied to each element. The purpose of specifying latent functions is to
    #plot them against the predictions.
    def define_latent_functions(self, functions, channels):
        self.latent_functions = {}
        for i in range(len(channels)):
            self.latent_functions[channels[i]] = functions[i]

    #Takes an X_prediction in the cumbersome format and computes the corresponding
    #y_prediction. It should be noted that X_prediction does not need to specify
    #all channels, just desired ones.
    def predict(self, X_prediction):
        self.X_prediction = X_prediction
        self.prediction_mean, self.prediction_std = self.model.predict_y(X_prediction)
        self.prediction_channels, self.elements_per_channel_prediction = self.determine_prediction_dims(X_prediction, self.prediction_mean, self.prediction_std)
        return self.prediction_mean, self.prediction_std

    #Creates X_prediction at given resolution (number of points) for the
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

    #REDO ABOVE TEXT TO REFLECT NEW WAYS
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

    def predict_interval(self, resolution, channels, start=None, end=None):
        starting_points = {}
        end_points = {}
        count = 0
        if (start != None):
            while(count < len(channels)):
                if (start[count] != None):
                    starting_points[channels[count]] = start[count]
                else:
                    starting_points[channels[count]] = self.find_border_points(self.X.get(channels[count]), 0)
                count = count + 1
        else:
            for channel in channels:
                starting_points[channel] = self.find_border_points(self.X.get(channel), 0)

        count = 0
        if(end != None):
            while(count < len(channels)):
                if (end[count] != None):
                    end_points[channels[count]] = end[count]
                else:
                    end_points[channels[count]] = self.find_border_points(self.X.get(channels[count]), 1)
                count = count + 1
        else:
            for channel in channels:
                end_points[channel] = self.find_border_points(self.X.get(channel), 1)

        X_predict = []
        input_dims = len(self.X.get(channels[0])[0])
        for channel in channels:
            linspaces = [np.linspace(starting_points.get(channel)[i], end_points.get(channel)[i], resolution) for i in range(input_dims)]
            counters = np.zeros(input_dims)
            for index in range(resolution**input_dims):
                point = [channel]
                for dim in range(input_dims):
                    point.append(linspaces[dim][int(counters[dim])])
                    counters[dim] = (counters[dim] + 1)%resolution
                X_predict.append(point)
        return np.array(X_predict)

    #Currently working for 1 feature in x axis (usually considered to be time).
    #If your input dimension is greater than 1 then you're going to have to resort
    #to other tricks to make your plots.
    def make_plots(self, savename, var=True, shape=None, dpi=300):
        if (self.Input_dims > 2):
            print("This is only defined for one or two dimensional inputs")
            return
        if(self.Input_dims == 2):
            self.make_plots_3d(savename, var, shape, dpi)
        sns.set(font_scale=3)
        with sns.axes_style("darkgrid"):
            sns.set_style("dark")
            # sns.set(rc={"font.size":20,"axes.labelsize":20})
            # plt.rcParams.update({'font.size': 22})
        #Uncomment the following 2 and comment the above 2 to change to dark theme
        # with sns.axes_style("dark"):
            # sns.set_style("dark", {"axes.facecolor": ".1"})
            if (shape == None):
                shape = [len(self.prediction_channels), 1]
                # fig, axes = plt.subplots(1,len(self.prediction_channels), figsize=(len(self.prediction_channels)*10 + 10, 10), sharey=False, constrained_layout=True, squeeze=False)

            fig, axes = plt.subplots(shape[0], shape[1], figsize=(shape[1]*10 + 10, shape[0]*10), sharey=False, constrained_layout=True, squeeze=False)
            #We suppose that X has one feature (Input_dim = 1)
            X_pred_one_feature = {}
            X_obs_one_feature = {}
            X_extra_one_feature = {}

            for channel in self.prediction_channels:
                x_pred = self.X_pred.get(channel)
                x_obs = self.X.get(channel)
                X_pred_one_feature[channel] = np.array([y[0] for y in x_pred])
                X_obs_one_feature[channel] = np.array([y[0] for y in x_obs])
                if(self.X_extra):
                    x_extra = self.X_extra.get(channel)
                    X_extra_one_feature[channel] = np.array([y[0] for y in x_extra])

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

                #We could compute these at plotting statements, but the more verbose approach was chosen for clarity.
                if(self.X_extra): #If extra observations are available.
                    valid_x_values_extra = np.where(np.logical_and(X_extra_one_feature.get(channel) >=x_min, X_extra_one_feature.get(channel) <=x_max)) #We take only those that fit within the prediction range.
                    X_to_plot_extra_data = X_extra_one_feature.get(channel)[valid_x_values_extra]
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
                    current_axis.plot(X_to_plot_extra_data, y_to_plot_extra_data, 'b.', ms=20) #Plot the extra observations, within prediction range.
                current_axis.plot(X_pred_one_feature.get(channel), self.mean_pred.get(channel), 'k-', lw=3) #Plot the posterior mean.
                current_axis.plot(X_to_plot, y_to_plot, 'r.', ms=20) #Plot the observations used to train the model, within prediction range.
                if(var == True):
                    current_axis.fill_between(X_pred_one_feature.get(channel), lower_variance_curve, upper_variance_curve, alpha=0.2) #Plot the posterior variance.
                if(is_latent_available == 1):
                    current_axis.plot(X_pred_one_feature.get(channel), latent_y, 'g--') #Plot the latent function for the prediction points, if it is available.
                current_axis.set_xlim([x_min - diff_x, x_max + diff_x]) #Modify the limits of each subplot.
                current_axis.set_ylim([y_min - diff_y, y_max + diff_y])
            plt.savefig(savename,dpi=dpi)
            plt.show()

    def make_plots_3d(self, savename, var, shape):
        pass

    #Returns posterior mean and std. Using this function only makes sense after
    #predictions.
    def get_prediction_info(self):
        return self.prediction_mean, self.prediction_std

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

    def determine_prediction_dims(self, X_pred, mean, std):
        channels, elements_per_channel = np.unique(X_pred[:,0], return_counts=True)
        self.X_pred = {}
        self.mean_pred = {}
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
            self.std_pred[channel] = std[start:end]
            start = end
            count = count + 1
        return channels, elements_per_channel

    #Prints the model parameters as a pandas table. Cannot be used before fit,
    #since GPFLow model has not been created before that point.
    def as_pandas_table(self):
        return self.model.as_pandas_table()

    #We can modify the stored observations, for a given channel.
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

    #In case the user does not want to compute the multioutput representation
    #this method takes 2 lists of numpy arrays or lists X and y. {X[i][j], y[i][j]} from j = 0
    #to j = X[i].shape[0] represent the i-th channel observations.
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

    #This function allows us to remove elements from a given channel.
    #elements_to_keep goes from 0 to 1, and represents the percentage
    #of kept elements, starting from the first. Defined for easy representation.
    def modify_channel(self, X, y, channel, start_percent, end_percent):
        if(start_percent < 0 or end_percent > 1 or end_percent < start_percent):
            print("Not a valid starting point")
            return X,y
        # elements = (X[channel].shape[0]-start)*elements_to_keep
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

        X_remaining = []
        for element in X_before:
            X_remaining.append(element)
        for element in X_after:
            X_remaining.append(element)

        y_remaining = []
        for element in y_before:
            y_remaining.append(element)
        for element in y_after:
            y_remaining.append(element)

        return X,y, X_remaining, y_remaining

    def remove_observations_at_random(self, X, y, ratio_of_deleted_items, number_of_centers):
        #multi-dim case is not trivially extended
        #multi-dim case has been trivially extended!
        if(ratio_of_deleted_items <= 0):
            print("No points are being removed")
            return X,y #?
        if(number_of_centers < 1):
            print("There must be at least 1 center")
            return #?
        if(ratio_of_deleted_items > 1.0):
            print("No points are being preserved")
            return #?
        removed_points_x = []
        removed_points_y = []
        slack = 0.0
        for channel in range(len(X)):
            total_to_delete = int(len(X[channel])*ratio_of_deleted_items)
            centerpoints, not_used_indexing = self.select_k_items(X[channel], number_of_centers)
            kept_observations, discarded_observations = self.kept_and_discarded_lists_per_center([X[channel],y[channel]], [[],[]], centerpoints[0], total_to_delete, slack)
            starting = 1
            while(len(discarded_observations[0]) < total_to_delete):
                for index in range(starting,len(centerpoints)):
                    kept_observations, discarded_observations = self.kept_and_discarded_lists_per_center(kept_observations, discarded_observations, centerpoints[index], total_to_delete, slack)
                slack = slack + 0.1
                starting = 0
            X[channel] = kept_observations[0]
            y[channel] = kept_observations[1]
            removed_points_x.append(discarded_observations[0])
            removed_points_y.append(discarded_observations[1])
        return X,y,removed_points_x, removed_points_y

    #Pick k items, uniformly, from a list in O(n) order
    def select_k_items(self, items, k):
        if(k > len(items)):
            print("k must be smaller than total element number.")
        reservoir = []
        indexing = []
        for index in range(k):
            reservoir.append(items[index])
            indexing.append(index)
        for index in range(k,len(items)):
            slot = random.randint(0, index)
            if (slot < k):
                reservoir[slot] = items[index]
                indexing[slot] = index
        return np.array(reservoir), np.array(indexing)

    def kept_and_discarded_lists_per_center(self, kept_pile, discard_pile, center, total_to_delete, slack):
        kept = []
        kept_y = []
        items = kept_pile[0]
        items_y = kept_pile[1]
        discard_pile_items = discard_pile[0]
        discard_pile_y = discard_pile[1]
        for index in range(len(items)):
            if(len(discard_pile_items) < total_to_delete):
                value = scipy.stats.bernoulli.rvs(scipy.stats.multivariate_normal.pdf(items[index],mean = center,cov = 0.2))
                if(isinstance(items[index], (list, np.ndarray))):
                    for i in range(len(items[index])):
                        if abs(items[index][i]-center[i]) > (0.2 + slack):
                            value = 0
                else:
                    if abs(items[index]-center) > (0.2 + slack):
                        value = 0
                if(value == 1):
                    discard_pile_items.append(items[index])
                    discard_pile_y.append(items_y[index])
                else:
                    kept.append(items[index])
                    kept_y.append(items_y[index])
            else:
                for i in range(index,len(items)):
                    kept.append(items[i])
                    kept_y.append(items_y[i])
                break
        return [kept, kept_y], [discard_pile_items, discard_pile_y]
