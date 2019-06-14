from itertools import chain
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from os import listdir
import time
from model import mogp_model

def conversion(label):
    if(label == 'bike'):
        return 0
    elif(label == 'climbing'):
        return 1
    elif(label == 'descending'):
        return 2
    elif(label == 'gymbike'):
        return 3
    elif(label == 'jumping'):
        return 4
    elif(label == 'running'):
        return 5
    elif(label == 'standing'):
        return 6
    elif(label == 'treadmill'):
        return 7
    elif(label == 'walking'):
        return 8

def get_file_names(path):
    folder = path + 'S0%d'
    filenames = []
    for x in range(1,10):
        current_folder = folder % x
        onlyfiles = [current_folder + "/" + f for f in listdir(current_folder)]
        filenames.append(sorted(onlyfiles))
    return filenames

def make_dataset(full_path):
    full_file_names = get_file_names(full_path)
    X = []
    y_names = []
    y_numbers = []
    for folder_number in range(9):
        for filename in range(len(full_file_names[folder_number])):
            path = full_file_names[folder_number][filename]
            sample_data = np.genfromtxt(path, delimiter=',')
            X.append(sample_data)
            label = path.split('/')
            label = label[len(label)-1].split('.')[0]
            label = label[0:len(label)-1]
            y_names.append(label)
            y_numbers.append(conversion(label))

    return X, y_numbers, y_names

def modify_channel(X,y,channel,elements_to_keep):
    elements = X[channel].shape[0]*elements_to_keep
    X_modified = X[channel][0:int(elements)]
    y_modified = y[channel][0:int(elements)]
    X[channel] = X_modified
    y[channel] = y_modified
    return X,y

full_path = '/home/mrlz/Desktop/MultiOutputSpectralMixture/data/HAR/Smartphone_Dataset/'
measurements, label_number, label_names = make_dataset(full_path)
#Small example for the first sample, corresponding to the signals of a bicycle ride.
#Since we're fitting curves, instead of performing a classification, we won't be using
#the class labels. We have to fabricate an X component to our y (the measurements).
#The measurements correspond to 9 channels (3 per sensor: accel, gyro, magnetometer) at
#a rate of 60hz. There's 500 measurements per channel, so the total time spanned is approx
#8.33s
X_list_bike = []
y_list_bike = []
X_list_climb = []
y_list_climb = []
measurements_for_one_bicycle_ride = measurements[0]
measurements_for_one_instance_of_climbing = measurements[5]
number_of_channels = 9
for index in range(number_of_channels):
    X_list_bike.append(np.array([x/60 for x in range(500)]))
    X_list_climb.append(np.array([x/60 for x in range(500)]))
    y_list_bike.append(measurements_for_one_bicycle_ride[:,index]-measurements_for_one_bicycle_ride[:,index].mean())
    y_list_climb.append(measurements_for_one_instance_of_climbing[:,index]-measurements_for_one_instance_of_climbing[:,index].mean())

starting_time = time.time()
model = mogp_model(4, optimizer = 'L-BFGS-B') #Powell takes a couple hours to compute, but always converges.
X_original, y_original = X_list_climb, y_list_climb


X_new, y_new, X_deleted, y_deleted = model.remove_slabs(X_list_climb, y_list_climb)
# X_input, Y_input = model.transform_lists_into_multioutput_format(X_new, y_new)
model.add_extra_observations(X_deleted, y_deleted, [i for i in range(number_of_channels)])


model.add_training_data(X_new,Y_new)

X_pred_new = model.predict_interval(1000, [x for x in range(number_of_channels)], start = [[0] for x in range(number_of_channels)],end = [[8.4] for x in range(number_of_channels)]) #First argument is the desired resolution (number of points between start and ending point), second argument are the desired channels to predict upon.
model.optimization_heuristic_zero()

mae = model.compute_mae(X_list_climb, y_list_climb)
print("MAE over original data = ", mae)
Y_pred, STD_pred = model.predict(X_pred_new)

print(model.read_trainables())
model.make_plots("smartphone_climbing_LBFGSB_10it_noise_slabs_9chan_with_mae.png", var=True) #Variance is included in the plot.
model.make_plots("smartphone_climbing_LBFGSB_10it_no_var_noise_slabs_9chan_with_mae.png", var=False) #Variance is not included.
# model.save("smartphone_model")
end_time = time.time()
print("Total time for climbing: ", end_time-starting_time)
