from model import mogp_model
import numpy as np
import csv
import time

def make_X_y_for_one_obj(data_of_object):
    #Since data_of_object are the rows pertaining the object
    #we merely need to represent the 6-channel info in the appropriate format
    #X has two values: [channel_id, mjd], where channel_id goes from 0 to 5
    #y has one value: [flux]
    X = np.zeros((data_of_object.shape[0],2))
    y = np.zeros(data_of_object.shape[0])
    counter = 0
    for row in data_of_object:
        X[counter,0] = row['band']
        X[counter,1] = row['mjd']
        y[counter] = row['flux']
        counter = counter + 1
    return X,y[:, None]

def make_X_y_matrices(path):
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        row_size = len(next(reader)) #This is the header row, no need to re-read it or add it to data
        size = (sum(1 for row in csvfile))
        #For now we'll be using float32, but if more precision is needed it can be changed to 64
        #We're using a structured array to benefit from np.sort's sorting by order, this will facilitate the separation of
        #time series per stellar object.
        data = np.zeros(size, dtype={'names':('id', 'mjd', 'band', 'flux', 'flux_err', 'detected'), 'formats': (int, np.float32, int, np.float32, np.float32, int)})
        counter = 0
        csvfile.seek(0)
        next(reader)
        for row in reader:
            data[counter][0] = row[0] #id
            data[counter][1] = row[1] #mjd
            data[counter][2] = row[2] #band
            data[counter][3] = row[3] #flux
            data[counter][4] = row[4] #flux_err
            data[counter][5] = row[5] #detected
            counter = counter + 1

        data = np.sort(data, order=['id', 'band', 'mjd']) #We sort each object by band and date, this allows us to plot easily in the model.
        X_y_of_each_object = []
        start = 0
        end = 0
        current_object = data[0]['id']
        for row in data:
            if(row['id'] != current_object):
                X_y_of_each_object.append(make_X_y_for_one_obj(data[start:end]))
                start = end+1
                current_object = row['id']
            end = end + 1
        X_y_of_each_object.append(make_X_y_for_one_obj(data[start:end])) #Last object will not find a different one
        return X_y_of_each_object

starting_time = time.time()
X_y_of_each_object = make_X_y_matrices('./data/PLAsTiCC/training_set.csv')
#Small example for the first stellar object of id 615
X, y = X_y_of_each_object[1]
# X_input, Y_input = model.transform_lists_into_multioutput_format(X, y)

model = mogp_model(5, optimizer = 'Powell') #Powell takes a couple hours to compute, but always converges.
model.add_training_data(X, y)
model.optimization_heuristic_zero()
X_pred_new = model.predict_interval(3200, [0,1,2,3,4,5]) #First argument is the desired resolution (number of points between start and ending point), second argument are the desired channels to predict upon.
Y_pred, STD_pred = model.predict(X_pred_new)
model.make_plots("PLAsTiCC_higher_res_novar_powell_10k_obj1_3200r.png", var=False) #Variance is not included in the plot because it is too high.
model.make_plots("PLAsTiCC_higher_res_var_powell_10k_obj1_3200r.png", var=True) #Variance is not included in the plot because it is too high.
end_time = time.time()
print("Total time: ", end_time-starting_time)
