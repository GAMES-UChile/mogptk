import mogptk
import numpy as np
import gpflow

# time array
n_points = 100
t = np.linspace(0, 6, n_points)

# data for channel 1
y1 = np.sin(6 * t)
# add noise
y1 += np.random.normal(scale=0.1, size=len(t))

# phased version
y2 = np.sin(6 * t + 2)
y2 += np.random.normal(scale=0.1, size=len(t))

# added sinosoidal
y3 = np.sin(6 * t) - np.sin(4 * t)
y3 += np.random.normal(scale=0.1, size=len(t))

# delayed and amplified
y4 = 3 * np.sin(6 * (t - 2))
y4 += np.random.normal(scale=0.1, size=len(t))

# data object for each channel
data1 = mogptk.Data(t, y1, name='A')

data2 = mogptk.Data(t, y2, name='B')

data3 = mogptk.Data(t, y3, name='C')

data4 = mogptk.Data(t, y4, name='D')

# create dataset
dataset = mogptk.DataSet(data1, data2, data3, data4)

# remove randomly
for data in dataset:
    data.remove_randomly(pct=0.3)

# remove for channel 0
dataset[0].remove_range(start=2.0, end=None)

# create model
model = mogptk.MOSM(dataset, Q=2)
# model = mogptk.CSM(dataset, Q=2)
# model = mogptk.SMLMC(dataset, Q=2)
# model = mogptk.CG(dataset, Q=2)

# initialize parameters of kernel
model.estimate_params()

print("ENTRENAR")
model.train(
    method='L-BFGS-B',
    tol=1e-60,
    maxiter=500)
print("FINISH")

x_pred = [t for i in range(len(dataset))]
model.predict(x_pred);
model.plot_prediction(title='Trained model');
