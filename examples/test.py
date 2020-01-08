# import library if it is not installed
import sys
sys.path.insert(0, '../')

import mogptk
import numpy as np
import gpflow


data = mogptk.DataSet()
data.append(mogptk.LoadFunction(lambda x: np.sin(5*x[:,0]), n=200, start=0.0, end=4.0, name='A'))
data.append(mogptk.LoadFunction(lambda x: np.sin(6*x[:,0])+2, n=200, start=0.0, end=4.0, var=0.03, name='B'))
data.append(mogptk.LoadFunction(lambda x: np.sin(6*x[:,0])+2 - np.sin(4*x[:,0]), n=20, start=0.0, end=4.0, var=0.03, name='C'))

data['A'].remove_random_ranges(3, 0.5)

data.set_pred_range(0.0, 5.0, n=200)

conv = mogptk.CONV(data, Q=3)
conv.print_params()

conv.estimate_params(method='SM')
conv.print_params()

conv.train()
conv.print_params()

exit(0)


mosm = mogptk.MOSM(data, Q=3)
mosm.print_params()

#mosm.estimate_params(method='SM')
#mosm.print_params()

mosm.train()
mosm.print_params()
